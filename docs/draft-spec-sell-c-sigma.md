# Draft spec: SELL-C-σ sparse format (USER-RULED 2026-07-10 — approved as POST-v1.0, nothing implemented yet)

Evaluate adding SELL-C-σ (Sliced ELLPACK with σ-window row sorting) as a **second** sparse
format next to BSR, targeting matrices with **no block structure** — the case where BSR
degenerates to `b=1` and its spMV runs as a scalar CSR loop that Burst cannot vectorize.
**The existing BSR format, its kernels, and `fProxyBSROperator` stay untouched.** This is a
complement, not a replacement (argued in §6).

Status: research + design draft for user review. No code exists. A hard benchmark gate
(§8, Stage 4) decides whether the feature ships or is parked.

---

## 0. Open questions for user

1. **Timing vs v1.0.** The release plan says the API is frozen for the public release. Is
   SELL-C-σ a post-v1.0 feature (my recommendation — it is purely additive but grows the
   public sparse surface), or do you want it in before the freeze becomes real?
2. **Is the unstructured-matrix customer real for you?** SELL-C-σ only pays off for
   matrices without dense b×b blocks: graph Laplacians / adjacency-like matrices, circuit
   matrices, scattered interpolation / RBF sparsifications, generic user triplets. If the
   target workloads are physics meshes (natural b∈{2,3,6} blocks) and the LP operators,
   this is a no-go regardless of the benchmark.
3. **ApplyT policy.** Recommended: match the BSR precedent — materialize `SELL(Aᵀ)` once
   (≈2× matrix memory) and route `ApplyT` through a forward SELL spMV, via a two-arg
   operator `fProxySELLOperator(in A, in AT)`. Alternative: a scalar scatter `spMVT` over
   the untransposed storage (no extra memory, no SIMD, write-conflict-serialized). Accept
   the 2× memory default?
4. **C fixed at 8?** I propose C = 8 as a compile-time constant (two `fProxy4` registers
   per chunk column — same shape as the library's proven 2-accumulator reduction pattern),
   identical for float and double. Alternative: C = 4 (one register). Stage 4 benchmarks
   both; the question is whether the *shipped* format fixes C (simpler, one code path) or
   exposes it.
5. **Naming.** New public names need approval: `fProxySELL` (struct), `fProxySELLOperator`,
   `arena.fProxySELL(...)`, `BSR.toSELL(...)` or builder-side `ToSELL(...)`, gallery
   `fProxyRandomSparseSkewed`. Bikeshed now, not during the build.
6. **Opt-in only?** Recommended: SELL is an explicit user choice (build it, wrap it in its
   operator, hand it to the same solvers). No auto-selection between BSR-b1 and SELL in
   the builder. Agree?

---

## 1. References (verified 2026-07-08, not from memory)

- **[Kreutzer2014]** M. Kreutzer, G. Hager, G. Wellein, H. Fehske, A. R. Bishop, *A Unified
  Sparse Matrix Data Format for Efficient General Sparse Matrix-Vector Multiplication on
  Modern Processors with Wide SIMD Units*, SIAM J. Sci. Comput. 36(5), C401–C423, 2014.
  DOI 10.1137/130930352; arXiv:1307.6209. Format definition, β metric, all numbers in §2/§4
  quoted from the arXiv full text.
- **[GHOST]** RRZE-HPC GHOST library (github.com/RRZE-HPC/GHOST): SELL-C-σ is its **only**
  sparse format, chosen so one layout serves CPU + Phi + GPU. Same group's newer
  Ultimate-SpMV (github.com/RRZE-HPC/Ultimate-SpMV) is also SELL-C-σ based — the format has
  a decade of production use, it is not a paper curiosity.
- Note: [Kreutzer2014] explicitly scopes to general matrices, single chip, forward spMV.
  **Transposed spMV is not addressed in the paper at all** — §5.3 below is our own analysis.

## 2. The format (as defined in [Kreutzer2014])

- Rows are sorted by descending nonzero count **within windows of σ consecutive rows**
  (σ=1 ⇒ no sorting), giving a row permutation P.
- The (sorted) matrix is cut into **chunks of C consecutive rows**. Each chunk i is padded
  to the length of its longest row, `cl[i]`, and stored **column-major within the chunk**:
  element (lane r, position j) of chunk i sits at `val[cs[i] + j*C + r]`.
- Arrays: `val[]` (padded values, zeros in the fill), `col[]` (matching column indices),
  `cs[]` (chunk start offsets), `cl[]` (chunk lengths). Plus P (and P⁻¹) when σ>1.
- spMV: for each chunk, C row-accumulators live in SIMD registers; the inner loop over
  j = 0..cl[i] does one contiguous `val`/`col` vector load, a gather of C entries of x, and
  one FMA. **No horizontal reduction, no reassociation** — each of the C rows is summed
  left-to-right in its own lane, in exactly the order a scalar CSR loop would use.
- **Chunk occupancy** β = Nnz / Σᵢ(C·cl[i]) measures padding waste (β=1 perfect,
  Eq. (6) worst case β→1/C for one dense row per chunk). Sorting exists to push β up:
  [Kreutzer2014] Table 2 reports, at C=16, σ=1→256: RM07R 0.63→0.93, kkt_power 0.54→0.92,
  webbase-1M 0.45→0.67; regular matrices (Hamrle3, ML_Geer) sit at β=1.00 with no sorting.
- Parameter guidance from the paper: **C as small as possible** while covering the SIMD
  width (they use C=4 on 256-bit AVX/double, C=16 on Xeon Phi, C=32 = warp on Kepler);
  σ a small multiple of C (σ=C² fixes their pathological case); **too-large σ backfires**
  by destroying x-vector locality (their kkt_power measurement degrades past σ≈2¹⁵).

## 3. Where this library actually stands (surveyed, not recalled)

- `fProxyBSR` (TemplateSource/Sparse/fProxyBSR.cs) is CSR-of-blocks; unrolled spMV kernels
  for square b∈{1,2,3,4,6} (UnsafeOP.Sparse.fProxy.cs). **`bsrMatVecB1` is a plain scalar
  CSR gather loop**: `y[br] += values[k] * x[colInd[k]]`. Under the library's FloatMode
  (no reassociation) Burst does not vectorize serial reductions — documented in
  Benchmarks/KernelBenchmark.cs and exploited everywhere via the 2×`fProxy4` accumulator
  pattern (8 independent lane-chains, UnsafeOP.fProxy.cs). That pattern **cannot** rescue
  CSR spMV: the x-gather is non-contiguous and the per-row sum is a single chain.
- SIMD widths: `fProxy4` codegens to `float4` / `double4` — **4 lanes for both dtypes**
  (128-bit float, 256-bit double). The proven fast shape is two `fProxy4` accumulators.
- `IfProxyLinearOperator` (Interfaces/LinearOperator.fProxy.cs) requires `Apply`, `ApplyT`,
  `ApplyBlock`. `ApplyT` consumers: `Krylov.cgls/lsqr/lsmr` + LS diagnostics. BSR already
  hit the transpose problem: scatter-based `spMVT` was the measured gap in cgls/lsqr
  (docs/features/sparse-bsr.md: ~7–8× vs the ideal ~14×), and the shipped fix is
  `arena.fProxyBSRTranspose(A)` + the two-arg `fProxyBSROperator(in A, in AT)`.
- Gallery (Sparse/Gallery.Sparse.fProxy.cs) has uniform-random fixed-expected-degree
  generators and a banded Laplacian2D. **Nothing with skewed row lengths** — i.e. no
  generator that exercises what σ-sorting is for; Stage 3 must add one or the benchmark lies.
- Permutation machinery exists (`Pivot` + application ops; `Indices` as the shared int
  buffer). `Pivot` is on the revised-simplex do-not-modify list — SELL uses `Indices` for
  its permutation and does its own gather at build time.

## 4. Where SELL-C-σ wins vs BSR (quantified for Burst)

**The one-sentence case:** SELL-C-σ is the only known layout that vectorizes spMV *without
reassociating any row sum*, which makes it the only SIMD spMV compatible with this
library's determinism stance — and the b=1 kernel it competes against is scalar.

Proposed Burst instantiation: **C = 8 = 2×`fProxy4`** for both dtypes. Inner loop per chunk
column position: two contiguous `fProxy4` loads of `val`, two of `col` (as `int4`), eight
scalar x-loads assembled into two `fProxy4` gathers, two FMAs into two register
accumulators. Eight independent accumulator chains per chunk — the same latency-hiding
shape as `vecDot`, but here each chain is a *different row*, so the result is bit-identical
to the scalar kernel and independent of C.

Expected spMV delta vs `bsrMatVecB1` on unstructured (blockless) matrices — estimates to be
confirmed/killed at the Stage 4 gate:

| Regime | Expected gain | Reasoning |
|---|---|---|
| In-cache (≲ L2/L3), β≈1 | **~1.5–3×** | b1 kernel is bound by the per-row FMA dependency chain (~4–5 cy latency) + scalar issue; SELL breaks it into 8 chains and streams val/col contiguously. Gathers of x (scalar loads) are the new limiter, which is why the ceiling is ~3×, not 8×. |
| DRAM-resident, β≈1 | **~1.0–1.5×** | Single-threaded Burst does not saturate DRAM, so some SIMD benefit survives (unlike [Kreutzer2014]'s socket-saturated CPU runs where SELL ≈ CRS); but at 8–12 B/nnz streamed the bandwidth ceiling compresses everything. |
| Any regime, β = 0.6 | multiply the above by ~β | Padding is streamed dead bandwidth and dead FMAs; this is exactly what σ-sorting recovers (Table 2 numbers in §2). |
| Blocked matrices vs BSR b∈{2,3,4,6} | **loss** | BSR amortizes one col index per b² values and its unrolled kernels are ~3× the general path; SELL pays 4 B of col index per nnz. SELL must never be pitched for blocked matrices. |

Honest translation notes from the paper's numbers: the headline 1.5–4× vs MKL-CRS is Xeon
Phi (16-wide) and cache-resident; on 8-core Sandy Bridge the multicore spMV is
memory-bound and SELL merely matches CRS. Our situation is *more* favorable than their CPU
case in one way (single thread ⇒ not bandwidth-saturated ⇒ compute wins count) and *less*
in another (4-lane registers, no hardware-gather guarantee through Burst — x gathers are
scalar load + insert).

## 5. Costs (the honest ledger)

### 5.1 Padding memory and bandwidth
Bytes/nnz: CSR-role BSR-b1 = 8 (float) / 12 (double) + row pointers. SELL = the same
divided by β, plus `cs/cl` (negligible) and P/P⁻¹ (8 B/row). Worst case β→1/C = 1/8 (one
dense row per chunk, σ too small); default σ = C² = 64 makes that construction impossible
within a window, and the paper's σ=256 recovered β≥0.9 on their nastiest RWC matrices.
**Rule: builder computes and reports β; documentation states SELL is a bad idea below
β≈0.7 — measure, don't guess.**

### 5.2 Row permutation bookkeeping
Design choice that removes most of the pain: **keep column indices unpermuted; only rows
are reordered.** Then `Apply` reads x directly (no input gather, x-locality identical to
CSR), and each chunk writes its C results through `y[P[row]]` — a scattered store per row
at chunk write-back, ~n stores per spMV, near-free next to nnz work. No solver, workspace,
or user vector ever sees permuted data; the permutation is an internal storage detail of
the struct. This forgoes the paper's option of solving entirely in permuted space (they
permute columns too) — acceptable because their measured locality risk (α blow-up) only
appears at σ≈2¹⁵ and we default σ=64. σ small ⇒ rows sharing x entries stay near each
other ⇒ bounded temporal-locality loss on x.

### 5.3 Transposed spMV — the cgls/lsqr question (taken seriously)
Column-major chunks make in-place `Aᵀx` a scatter: at chunk column j, the C lanes want
`y[col[...]] += val[...] * x[lane-row]`, and **two lanes in the same vector may target the
same y entry** — SIMD requires conflict detection; correctness requires serialization. So
native transposed SELL spMV is scalar-at-best, plus it reads x through P. It is genuinely
awkward — but it is *not* the dealbreaker, because the library already solved this exact
problem for BSR: **materialize the transpose once, per matrix, and give the operator both.**
`SELL(Aᵀ)` gets its own σ-sort over Aᵀ's row lengths (= A's column degrees) and its own
internal permutation; `ApplyT` becomes a forward SELL spMV with all of §4's speedup. Costs:
≈2× matrix memory (precedented; `fProxyBSRTranspose` does the same) and Aᵀ's β may differ
from A's (a tall LS matrix with skewed *column* degrees pads differently — report both
βs). A scalar fallback `spMVT` over the untransposed storage ships anyway (Stage 1) for
memory-constrained users, mirroring BSR's one-arg operator. Verdict: with the two-arg
operator pattern, cgls/lsqr/lsmr work unchanged and get faster, not slower; without it,
SELL would indeed be a net loss for LS solvers. The pattern is the design.

### 5.4 Construction
From the existing builder triplets (or from a finished BSR-b1): count row lengths O(nnz),
stable-sort rows by length within σ-windows O(n·log σ) using `Indices`, compute `cl/cs`,
one gather pass to fill padded `val/col`. Same order of work as `ToBSR` compression.
Padding entries get `val=0` and a **safe in-range column index** (the row's own first
column; row 0's col 0 for empty rows) — see NaN caveat in §9.

### 5.5 API surface
One struct, one operator, one builder method, one gallery generator, benchmarks, docs —
all additive. The real cost is a permanent second sparse format to maintain and explain
("BSR for blocked, SELL for blockless" must be one sentence in the docs, and is).

## 6. Fit: complement, not replacement

Replacement is a non-starter on three grounds: (1) BSR's blocked kernels amortize index
storage b²-fold and are already ~3× the general path — SELL structurally cannot match that
on blocked matrices (per-nnz col index, no dense-block inner kernels); (2) `Symmetric`
BSR's storage-halving has no SELL analogue (SELL-transpose-in-place is the §5.3 scatter
problem); (3) the LP stack (`fProxyNormalOperator`, LAD/slack operators) composes over BSR
and is freshly tuned — churning its storage for zero expected win is pure risk. GHOST could
afford SELL-only because it targets bandwidth-saturated HPC nodes where formats converge;
we are single-threaded and blocked matrices are a headline use case.

Who benefits: users assembling **blockless** matrices from triplets — graph
Laplacians/adjacency (power-law degrees are the σ showcase), circuit/network matrices,
scattered-data LS (tall, feeds cgls/lsqr via the two-arg operator). Who does not: physics
meshes with natural b∈{2,3,6} DOF blocks (BSR), the LP interior-point/PDLP operators
(keep BSR), small or dense-ish matrices, and anyone whose matrix is uniform-row-length
*and* DRAM-resident (gain compresses toward 1× — §4 table).

Integration: `fProxySELLOperator : IfProxyLinearOperator` (Apply / ApplyT / ApplyBlock =
row-loop over Apply initially) — **every existing solver (cg/pcg/minres/cgls/lsqr/lsmr,
eigensolvers) works unchanged, zero solver edits.** This is the payoff of the operator
abstraction and the whole reason the feature is cheap to bolt on.

## 7. Recommendation

**Qualified GO — as a complement, post-v1.0, behind a kill-gate benchmark.**

Decision criteria (stated so the user can veto any leg):
1. *Customer exists* (open question 2). If no blockless-matrix workload matters, NO-GO.
2. *Kill gate*: Stage 4 must measure **≥1.5× spMV vs BSR-b1** in-cache on unstructured
   matrices at β≥0.9, and **no regression vs BSR-b1 beyond 10%** in any measured cell
   (DRAM-bound, low-β) that the docs don't explicitly disclaim. Miss ⇒ park the branch,
   write the numbers into this doc, done — the research still retires the question.
3. *No solver churn*: solvers and BSR untouched; SELL enters only through the operator.
4. *Determinism preserved*: SELL spMV bit-identical to the scalar CSR reference per row
   (fixed left-to-right order per lane) — this is a test, not a hope.

Why GO at all: BSR-b1 is the library's answer for blockless matrices today, and it is
scalar under a FloatMode that forbids the compiler from fixing it; SELL-C-σ is the
standard, production-proven (GHOST) layout whose entire design goal is SIMD-vectorizing
exactly that case without reordering any row sum. It slots into the operator interface
with zero solver changes. Why qualified: 4-lane registers and scalar gathers cap the win
well below the paper's Phi/GPU headlines, and only a benchmark can say whether 1.5–3×
survives contact with Burst.

## 8. Staged plan (if GO; each stage = one coder-agent round, suite green both dtypes before the next)

- **Stage 1 — format + reference kernel.** `fProxySELL` struct (dual-mode arena/standalone,
  mirroring `fProxyBSR`'s record pattern), construction from `fProxyBSRBuilder` triplets
  and from an existing BSR-b1 (`toSELL(sigma)`), β computed and stored, scalar reference
  spMV + scalar scatter spMVT in `UnsafeOP`. Direct kernel tests vs dense reference;
  known-failure tests (empty rows, empty matrix, NaN-in-x padding case §9, σ=1, C∤n tail
  chunk); a literature-anchored known-answer matrix. No SIMD yet.
- **Stage 2 — SIMD kernel + operator.** C=8 chunk kernel (2×`fProxy4`), bit-identity test
  vs Stage 1 scalar kernel (criterion 4). `arena.fProxySELLTranspose(A)`; one-arg and
  two-arg `fProxySELLOperator`; cg/cgls/lsqr/lsmr integration tests on the same matrix as
  a BSR operator, solutions matched to tolerance.
- **Stage 3 — gallery.** `fProxyRandomSparseSkewed` (power-law/Zipf row degrees, seeded,
  O(nnz), SPD and rectangular variants) — the honest test/benchmark matrix class that the
  gallery currently lacks; β assertions (σ=1 low, σ=C² high) as tests.
- **Stage 4 — benchmark + kill gate.** Per docs/spec-shipped-feature.md harness (IJob.Run,
  1 warmup + 4 timed, median, results via output arrays): SELL vs BSR-b1 vs dense, sizes
  in/out of cache, fills {1%, 7%, 33%}, uniform vs skewed rows, σ∈{1, C², 256}, C∈{4,8}
  once to settle question 4, plus cgls/lsqr end-to-end vs BSR two-arg operator. Apply
  criterion 2; write the verdict and numbers back into this spec; only then docs/README.

Files (templates are source of truth): `TemplateSource/Sparse/fProxySELL.cs`,
`fProxySELLOperator.cs`, additions to `SparseOP.fProxy.cs`, `UnsafeOP.Sparse.fProxy.cs`,
`Gallery.Sparse.fProxy.cs`, `Arena.Sparse.fProxy.cs`; tests
`Tests/.../SparseSELLTests.fProxy.cs`; bench section in the sparse benchmark file.

## 9. Numerics & edge notes

- **Padding vs NaN/Inf in x**: padded entries compute `0·x[padCol]`; if x is non-finite at
  `padCol` this injects NaN into a row that logically never reads it. Pointing `padCol` at
  the row's own first column confines pollution to the row's true read set; the residual
  hole is **empty rows** (padCol=0 by convention): all-zero row + non-finite x[0] yields
  NaN instead of 0. Document + pin with a test (integer-surface-policy precedent: document
  pathologies, don't build machinery).
- Determinism: per-row left-to-right accumulation in a dedicated lane ⇒ bit-identical to
  scalar CSR order, independent of C and σ; padding adds exact `+0.0` terms (sign-of-zero:
  `-0.0` results become `+0.0` only if a row is all `-0.0` products — below the library's
  documented tolerance bar, note it in the struct doc).
- σ default 64 (=C²), builder parameter; σ=1 supported (= plain Sliced ELLPACK, no
  permutation arrays allocated).

## 10. Rejected alternatives (one line each)

- **Plain CSR**: already exists in effect as BSR-b1; same scalar non-vectorizable kernel,
  so adding it buys nothing.
- **ELLPACK**: = SELL-N-1; β collapses toward 1/N-ish on any skewed matrix ([Kreutzer2014]
  Eq. (6) pathology) — strictly dominated by SELL-C-σ.
- **Sliced ELLPACK (no sorting)**: is exactly SELL-C-1; supported for free via σ=1, not a
  separate format.
- **CSR5** (Liu & Vinter 2015): SIMD via segmented sums that *reassociate* row totals —
  breaks the bit-identical-to-scalar guarantee and its tile-transpose machinery is heavy;
  wrong trade for this library.
- **JDS/pJDS**: the GPU ancestors of SELL-C-σ (global sort / padded jagged diagonals);
  subsumed by SELL-C-σ per the paper itself.
- **Autotuned register blocking (OSKI-style)**: targets block structure — BSR already owns
  that ground with hand-unrolled kernels.

## 11. Decisions (USER-RULED 2026-07-10)

1. **Timing: POST-v1.0** (user accepted the recommendation). Do not start before the release.
2. **Customer: confirmed in principle** — user independently identified the b=1 case as
   SELL's domain ("block matrices of size 1 don't make sense, that's why I was thinking of
   sell-c"); the roadmap's heat/fluids demos are the concrete workloads.
3. **All §0 defaults accepted**: ApplyT via one-time materialized SELL(Aᵀ) (~2× memory,
   BSR precedent); C=8 fixed (2×fProxy4), not exposed; strictly opt-in (no auto-selection);
   names as proposed (fProxySELL / fProxySELLOperator / arena.fProxySELL / BSR.toSELL).
4. Kill-gate (§7 criterion 2) unchanged and binding when the build starts.
