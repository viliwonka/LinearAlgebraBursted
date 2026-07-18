# Spec: one-level Additive Schwarz / Restricted Additive Schwarz preconditioner over BSR

Status: specced (Fable, 2026-07-18). Not implemented. One-level MVP — no coarse space. Every
unilateral design call is listed in "Open questions for the user" at the bottom for veto.

User ask: "a generic preconditioner better than IC(0)", via domain decomposition ("domain
separation"). Deliverable: overlapping-domain-decomposition preconditioners — symmetric Additive
Schwarz (AS) for `pcg`/`pminres`, Restricted Additive Schwarz (RAS) for `pbiCGStab` — as drop-in
`IfProxyPreconditioner` implementations over `fProxyBSR`, with the same arena-factory + typed
overload-ladder conventions as `fProxyBlockJacobi` / `fProxySSOR` / `fProxyIC0` / `fProxyILU0`.

## What exists (survey, 2026-07-18)

- Interface: `IfProxyPreconditioner` (`Interfaces/LinearOperator.fProxy.cs:54`) — single method
  `void Apply(in fProxyN r, ref fProxyN z)`; z must not alias r; no Rows/Cols on the interface,
  implementations validate against their own stored shape. Solvers are generic over
  `TPre : struct, IfProxyPreconditioner`.
- Existing preconditioners (all in `TemplateSource/Sparse/`):
  - `fProxyBlockJacobi` — readonly struct, own arena record table (`fProxyBlockJacobiRecord`),
    dual-mode inline/arena storage, `IDisposable`, per-block LU/Gauss-Jordan inverses cached at
    build, `Apply` = block-diagonal matvec with unrolled BR ∈ {1,2,3,4,6} fast paths.
  - `fProxySSOR` — readonly struct composed of arena-tracked pieces; mirrors Symmetric-storage A
    to full via `fProxyBSRMirrorToFull` at setup.
  - `fProxyIC0` — readonly struct "composed entirely of arena-tracked pieces — no record table of
    its own, no Dispose()"; Manteuffel escalating diagonal-shift retry on breakdown (6 attempts,
    `1e-3·diagMax` then ×10); non-throwing ctor twin with `out PreconditionerInfo`.
  - `fProxyILU0` — the nonsymmetric sibling of IC0, used only by `pbiCGStab`. This IC0/ILU0 split
    is the exact precedent for the AS/RAS split below.
- Arena factories (`Sparse/Arena.Sparse.fProxy.cs`): `arena.fProxyIC0(in A)` +
  `arena.fProxyIC0(in A, out PreconditionerInfo info)` twins; same for BlockJacobi/SSOR/ILU0.
- Overload ladders: `pcg` and `pminres` each have THREE rungs per preconditioner type
  (zero-alloc with explicit `ref r, p, Ap, z` scratch + `maxIter, tol`; temp-vec-allocating +
  `maxIter, tol`; defaults `A.M_Rows` / `Consts.fProxySqrtEps`), forwarding into the generic
  `pcg{TOp,TPre}` via `fProxyBSROperator`. `pbiCGStab` has TWO rungs for `fProxyILU0`
  (temp-allocating + defaults; `Krylov.PBiCGStab.fProxy.cs:115,132`).
- BSR (`Sparse/fProxyBSR.cs`): RowPtr/ColInd/Values per block; blocks within a block-row stored
  in ascending ColInd (invariant); block interior row-major; `Symmetric=true` = lower block
  triangle only (requires BR==BC, square grid); `Nnzb`; arena-tracked with generation stamps.
- Dense factorizations: `CHO.decompInPlace(ref fProxyMxN A_to_L)` → `DirectSolveInfo`, plus
  `decompSolve` for solve-against-cached-factor (4-token grid per solver-api-rework);
  `LU.decompInPlace(ref fProxyMxN, ref Pivot)` + `LU.decompSolve` (used by BlockJacobi's BR>16
  path). `PreconditionerInfo { DirectSolveStatus status; double shift; int attempts; }`.
- Gallery (`Sparse/Gallery.Sparse.fProxy.cs`): `fProxyLaplacian2D(gridX, gridY)` (Poisson, BR=1),
  `fProxyPenalizedGrid3D(nx, ny, nz, EA, penalty)` (elasticity/truss-frame, penalty-conditioned),
  `fProxyRandomSparseSPD`, `fProxyRandomSparse`.
- Warm-state IJob lesson (memory: job-struct-copy audit, LOBPCG cache-copy bug): mutable state
  that must survive an IJob struct-copy MUST live behind native memory (Arena/NativeArray), never
  in struct scalar fields mutated after construction. Test preconditioned solves THROUGH an IJob.

## Design decisions

### D1. Two structs: symmetric AS and RAS — CG safety via the type system (CRITICAL)

Math: with `R_i` the restriction onto overlapped subdomain i and `A_i = R_i A R_iᵀ`:

- **Additive Schwarz (AS)**: `M⁻¹ = Σ_i R_iᵀ A_i⁻¹ R_i`. SYMMETRIC (and SPD — see D4) →
  valid for `pcg` and `pminres`.
- **Restricted AS (RAS)**: `M⁻¹ = Σ_i R̃_iᵀ A_i⁻¹ R_i`, where `R̃_i` restricts onto the
  NON-overlapped partition cell (Cai & Sarkis 1999). NON-SYMMETRIC even for SPD A — typically
  ~20-30% fewer iterations than AS and no duplicate-scatter work, the workhorse for
  GMRES/BiCGStab — but it silently breaks CG's convergence theory (CG requires an SPD
  preconditioner; RAS-preconditioned CG can stagnate or diverge with no error raised).

Decision: **two public structs, mirroring the IC0/ILU0 precedent**:

- `fProxyAdditiveSchwarz` (symmetric AS) — gets `pcg` and `pminres` ladder rungs.
- `fProxyRestrictedSchwarz` (RAS) — gets `pbiCGStab` ladder rungs ONLY.

The typed convenience ladder makes misuse impossible without going through the generic
`pcg{TOp,TPre}` escape hatch (which is already caveat-emptor for every `TPre`). Both structs'
XML docs state the contract in one line each: AS "symmetric — valid for pcg/pminres"; RAS
"NON-symmetric — never use with pcg/pminres/CG; use pbiCGStab".

Rejected alternatives:
- One struct + a `bool restricted` option: a RAS-built instance would satisfy the `pcg(in A, in
  fProxyAdditiveSchwarz M, ...)` rung and defeat the type guard. Rejected.
- Multiplicative Schwarz (Gauss-Seidel over subdomains): better convergence per sweep but
  sequential across subdomains, nonsymmetric unless doubled (symmetrized sweep = 2× apply cost),
  and blocks any future parallel Apply. Rejected for MVP; note as a possible variant later.
- RAS-with-harmonic-gather (ASH, `R̃` on the gather side instead): no clear win over RAS;
  rejected.

Local factorization differs per struct (see D4): AS uses dense Cholesky (`CHO`), RAS uses dense
LU with partial pivoting (`LU` + `Pivot`) since its target is general square A.

### D2. Partitioning: contiguous block-row ranges (MVP), fixed-order greedy BFS later

Subdomains partition the BLOCK-row index space `0..BlockRows-1` (never split a block — a block is
BR coupled scalar unknowns).

MVP: **contiguous ranges**. `K = ceil(BlockRows / blocksPerSub)` subdomains; subdomain i owns
block-rows `[i·blocksPerSub, min((i+1)·blocksPerSub, BlockRows))`. `blocksPerSub =
max(1, opts.subdomainSize / BR)` where `subdomainSize` is in SCALAR unknowns (default 128,
sane range ~32-256; see D4 for why not larger).

- Trivially deterministic, zero graph work, and effective whenever the block numbering has
  locality — which holds for every gallery generator (lexicographic grid numbering) and for
  typical FEM/truss assembly orders.
- Weakness: ignores connectivity; on a scrambled numbering the subdomains are unions of
  disconnected fragments and AS degrades toward block-Jacobi-with-overlap (still correct, still
  SPD, just weaker).

Phase-2 alternative (spec'd, not MVP): greedy BFS aggregation over the block-adjacency graph —
seed at the lowest-indexed unassigned block-row, BFS in ascending-neighbor order (BSR's
ascending-ColInd invariant gives a canonical neighbor order for free) until the size target,
repeat. Deterministic (fixed seed rule + fixed visit order), gives connected subdomains on any
numbering. Costs an extra O(nnzb) pass and a visited/queue workspace. Same options struct grows a
`partition` enum then; not before.

### D3. Overlap: δ layers of block adjacency; R_i and R̃_i

Each subdomain's OWNED block set `Ω_i` (the contiguous range, a true partition) is extended by δ
layers of the block-adjacency graph to the OVERLAPPED set `Ω_i'`:

- Layer step: `Ω ← Ω ∪ { j : block (r,j) or (j,r) stored for some r ∈ Ω }`. Adjacency must be
  symmetrized: for `Symmetric=true` BSR only the lower triangle is stored, and even full-storage
  patterns may be structurally unsymmetric (RAS/general A). Setup builds a transient symmetrized
  adjacency (or mirrors via `fProxyBSRMirrorToFull` on `Allocator.Temp`, SSOR precedent, disposed
  before setup returns — A itself is only read at setup, never at Apply).
- δ default = 1; δ = 0 legal (degenerates to non-overlapping block solves = a coarse-grained
  block-Jacobi; AS==RAS then). δ tunable via options.
- `Ω_i'` is stored SORTED ASCENDING by global block index — the canonical local ordering. With
  contiguous partitioning the owned blocks are exactly the members of `[lo_i, hi_i)`, so the
  "owned" predicate is a range check on the global index — no mask storage needed in the MVP.

Operators, concretely:
- `R_i` (gather): copy the BR scalars of each block in `Ω_i'`, in ascending global order, into a
  local vector of length `n_i = |Ω_i'|·BR`.
- `R_iᵀ` (AS scatter): add each local entry back to its global slot. Overlapped dofs receive
  contributions from every subdomain whose `Ω_i'` contains them — this summation is what makes
  AS symmetric.
- `R̃_iᵀ` (RAS scatter): write back ONLY the entries whose block is in the owned range `Ω_i`.
  Since `{Ω_i}` is a partition, every global dof is written by exactly one subdomain — no
  summation at all.

### D4. Local solve: gather-once, dense-factor-once at setup; cached factors; memory formula

Setup, per subdomain i (fixed order i = 0..K-1):

1. Gather the local dense matrix `A_i` (`n_i × n_i`): for each block-row r ∈ `Ω_i'`, walk its
   ColInd (ascending) and copy blocks whose column ∈ `Ω_i'` into the dense local matrix at the
   mapped local offsets. Global→local block map = one `Indices` workspace of length BlockRows
   (stamped per subdomain, or filled −1 and reset), reused across subdomains. For
   `Symmetric=true` A, gather from the temp full mirror (D3).
2. Factor in place: AS → `CHO.decompInPlace`; RAS → `LU.decompInPlace(ref, ref Pivot)`.
   On CHO breakdown, retry with the IC0-style Manteuffel escalating diagonal shift on the LOCAL
   matrix (same schedule: `1e-3·diagMax`, ×10, 6 attempts). Note: if A is exactly SPD every
   principal submatrix is SPD, so shifts only fire on numerical breakdown (float roundoff,
   near-semidefinite penalty limits) — and whenever ALL local factors succeed (shifted or not),
   `M = Σ R_iᵀ (A_i + α_iI)⁻¹ R_i` is symmetric positive definite BY CONSTRUCTION (sum of SPSD
   terms whose ranges jointly cover ℝⁿ, since `∪Ω_i' ⊇ ∪Ω_i` = everything). That is the clean
   contract that makes AS always CG/MINRES-safe once the build reports Success.
3. Store the factor (and Pivot, RAS) into the flat arena-owned factor buffer at its offset.

`PreconditionerInfo` aggregation across subdomains: `status` = Success iff every local factor
succeeded, else the first failing subdomain's status; `shift` = max local shift used; `attempts`
= max local attempts. Non-throwing `out info` twin + throwing ctor, exactly like IC0.

Costs (N = scalar dims, s = `subdomainSize`, g = overlap inflation `n_i/(s)` ≈ 1 + O(δ ·
surface/volume of a subdomain), K ≈ N/s):

- Setup flops: `Σ n_i³/3 ≈ N·g³·s²/3` — heavy (s=128, g=1.5 → ~1.8e4 flops per unknown).
  Setup-once, amortized over many Applies; this is the intended factor-cache usage exactly like
  BlockJacobi/IC0, only bigger.
- Factor memory: `Σ n_i² ≈ N·g²·s` scalars. s=128, g=1.5, N=65k, double → ~150 MB — REAL.
  This is why `subdomainSize` caps around ~256 and why the spec keeps full-square (not packed)
  storage only for the MVP (packed triangular would halve it; open question). The XML doc and
  the options doc must state the `O(N·g²·s)` memory formula so users size consciously.
  Geometry note: g is worst for thin subdomains (a 128-wide strip of a 2D grid with δ=1 gathers
  two full neighbor lines → g≈3); square-ish aggregates (Phase-2 BFS) shrink g.
- Apply flops: `Σ 2·(n_i²/2) ≈ N·g²·s` per call (forward+backward sweep, or pivoted LU solve) —
  for s=128 roughly 10-30× the flops of an IC0 apply (≈ 2·nnz). Dense triangular sweeps are
  branch-free and SIMD/prefetch-friendly, but the honest framing is: AS must cut ITERATIONS by
  more than its per-iteration cost to win wall-clock; its guaranteed win is robustness +
  iteration count (see Tests).

Rejected alternatives:
- Sparse local factorization (per-subdomain IC(0)/ILU(0) on the local pattern): kills the memory
  blowup but reintroduces incompleteness inside subdomains, gutting the "strong local solve"
  advantage that justifies Schwarz over IC0 at all. Rejected for MVP; the memory-pressure escape
  hatch is "use a smaller `subdomainSize`", and the real fix is the two-level upgrade (D6).
- Explicit local inverses (BlockJacobi-style `DInv`): same memory, worse setup cost (n³ vs n³/3),
  and matvec-apply is not faster than two triangular sweeps at these sizes. Rejected.

### D5. Apply: gather → cached-factor solve → scatter, fixed subdomain order

`Apply(in r, ref z)`, both structs:

```
z ← 0                                  // AS only; RAS writes every dof exactly once, no clear
for i = 0..K-1:                        // fixed ascending order
    gather   rLoc[0..n_i) ← r at Ω_i' (ascending global order)
    solve    x_i = A_i⁻¹ rLoc          // CHO/LU triangular sweeps against the CACHED factor
    scatter  AS:  z[Ω_i'] += x_i       // overlapped sum
             RAS: z[Ω_i]   = x_i restricted to owned entries (range check on global block index)
```

- Aliasing: z must not alias r (r is re-gathered by later subdomains after z is partially
  written) — same guard and exception style as BlockJacobi/IC0.
- Scratch: one arena-owned `max n_i`-length local vector, allocated at setup, reused per
  subdomain (buffer CONTENTS mutate during Apply; no struct field does — see IJob section).
  Consequence: a single preconditioner instance is not safe for concurrent Apply from multiple
  threads. Existing solvers are single-threaded; document, don't guard.
- The local triangular solve runs against the cached factor via the `decompSolve` token
  (CHO / LU-with-Pivot) operating on the factor slice — no refactorization ever happens in
  Apply.
- Future parallel Apply (NOT MVP): RAS's scatter is conflict-free by construction
  (partition ⇒ each dof written once) so subdomain-parallel RAS is deterministic with zero
  atomics; symmetric AS's overlapped `+=` would need an ordered reduction to stay deterministic.
  Worth a DEVLOG note, not code.

### D6. Scope: one-level only; two-level coarse space is the upgrade path

One-level AS/RAS is NOT algorithmically scalable: with fixed subdomain size, iteration count
grows roughly like `O(1/H)` ~ K^(1/d) (no global information transfer per apply — condition
number `κ(M⁻¹A) = O(1/(H·δ_geom))`). For the target problem sizes here (10³-10⁵ unknowns,
tens-to-hundreds of subdomains) that growth is mild and one-level AS still beats one-level IC(0)
where it matters, but the spec states it plainly: the fix is a COARSE space (two-level AS —
`M⁻¹ += R_0ᵀ A_0⁻¹ R_0` with A_0 a Galerkin coarse operator), which is the same machinery as
`docs/dev/spec-multigrid-solver.md` (unsmoothed aggregation + rigid-body near-nullspace,
deterministic segmented-reduction RAP). If/when that spec's MVP lands, its tentative prolongator
is exactly the coarse space this preconditioner needs — implement two-level Schwarz as a thin
composition then, don't build a private coarse space here.

## API surface

New shared options type (no proxy tokens — one emitted file, `PreconditionerInfo` precedent):

```csharp
/// <summary>Options for the Schwarz preconditioners. subdomainSize is the target subdomain size
/// in SCALAR unknowns (rounded down to whole blocks, min one block); overlap is the number of
/// block-adjacency layers added to each subdomain. Factor memory scales as O(N·overlapFactor²·
/// subdomainSize) — see the struct docs.</summary>
public struct SchwarzOptions
{
    public int subdomainSize;   // default 128
    public int overlap;         // default 1; 0 legal
    public static SchwarzOptions Default => new SchwarzOptions { subdomainSize = 128, overlap = 1 };
}
```

(`int` fields default fine — the CS1750 proxy-default-param trap does not apply.)

Arena factories (`Arena.Sparse.fProxy.cs`, mirroring the IC0/ILU0 quartet):

```csharp
public fProxyAdditiveSchwarz  fProxyAdditiveSchwarz(in fProxyBSR A, in SchwarzOptions opts);
public fProxyAdditiveSchwarz  fProxyAdditiveSchwarz(in fProxyBSR A, in SchwarzOptions opts, out PreconditionerInfo info);
public fProxyAdditiveSchwarz  fProxyAdditiveSchwarz(in fProxyBSR A);                    // Default opts
public fProxyRestrictedSchwarz fProxyRestrictedSchwarz(in fProxyBSR A, in SchwarzOptions opts);
public fProxyRestrictedSchwarz fProxyRestrictedSchwarz(in fProxyBSR A, in SchwarzOptions opts, out PreconditionerInfo info);
public fProxyRestrictedSchwarz fProxyRestrictedSchwarz(in fProxyBSR A);
```

Contract lines for the ctor docs: A square (`BlockRows==BlockCols`, `BR==BC`); Symmetric-storage
A accepted (mirrored transiently at setup); arena-owned, disposed with the arena; build outcome
via the out-info twin (throwing ctor otherwise, IC0 wording).

Solver ladder rungs (EXACTLY the existing per-type pattern, no new shapes):

- `Krylov.fProxy.cs` — `pcg(in fProxyBSR A, in fProxyAdditiveSchwarz M, ...)` × 3 rungs
  (zero-alloc `ref r,p,Ap,z` + maxIter/tol; temp-alloc + maxIter/tol; defaults).
- `Krylov.PMinres.fProxy.cs` — `pminres(..., in fProxyAdditiveSchwarz M, ...)` × 3 rungs.
- `Krylov.PBiCGStab.fProxy.cs` — `pbiCGStab(in fProxyBSR A, in fProxyRestrictedSchwarz M, ...)`
  × 2 rungs (temp-alloc + defaults, matching the ILU0 rungs).

No `pcg`/`pminres` rung takes `fProxyRestrictedSchwarz` — that absence IS the CG-safety
contract (D1). Whether `pbiCGStab` should ALSO get `fProxyAdditiveSchwarz` rungs (legal, just
slower than RAS) is an open question; default no, to keep the ladder minimal per the
iterative-solver-overload-ladder ruling.

## Data layout and Arena allocation

Both structs are `readonly struct : IfProxyPreconditioner`, composed entirely of arena-tracked
pieces (IC0 model — no record table of their own, no Dispose; everything dies with the arena).
Shared internal builder; the structs differ only in factor kind and scatter.

Per-preconditioner state (all fixed at setup, sizes known after the overlap pass):

| field | type (arena-owned) | contents |
|---|---|---|
| `SubStart` | `Indices`, len K+1 | prefix offsets into `SubBlocks` (CSR-of-subdomains) |
| `SubBlocks` | `Indices`, len Σ\|Ω_i'\| | each subdomain's overlapped block ids, ascending |
| `OwnedLo/OwnedHi` | `Indices`, len K each | owned block-range per subdomain (RAS scatter + tests) |
| `FactorStart` | `Indices`, len K+1 | prefix offsets into `Factors` (entries = Σ n_i²) |
| `Factors` | `fProxyN` (flat) | cached local CHO L / LU factors, row-major per subdomain |
| `Piv` | `Indices`, len Σ n_i | RAS only: per-subdomain pivot rows |
| `Scratch` | `fProxyN`, len max n_i | Apply-time local vector (contents mutable) |
| readonly ints | struct fields | `BlockRows`, `BR`, `K`, `MaxLocalN`; `Rows => BlockRows*BR` |

`Indices` is the shared int-vector type (codegen float-in/int-out rule: reuse Pivot-style shared
types, never hand-duplicate an int container). Setup-transient allocations (symmetrized
adjacency / full mirror, global→local map, BFS workspace later) are `Allocator.Temp` or
arena-then-logically-dead — coder's call, but nothing transient may survive into Apply.

## Determinism

- Partition: pure integer arithmetic. Overlap: layer expansion visits block-rows ascending and
  neighbors in stored (ascending-ColInd) order; the resulting `Ω_i'` sets are sorted — fully
  deterministic, platform-independent.
- Setup factorization: dense CHO/LU touch only `+ - * / sqrt` in fixed loop order →
  cross-arch deterministic under FloatMode.Strict per docs/dev determinism analysis (same class
  as the existing direct solvers). LU partial pivoting compares magnitudes — deterministic
  tie-break = first max (existing LU behavior).
- Apply: subdomains processed 0..K-1; gathers/solves fixed-order. AS's overlapped `+=` has a
  FIXED summation order (ascending subdomain id at each dof) → bit-reproducible. RAS's scatter
  is a partition — each dof written exactly once — order-independent by construction.
- Same-input same-arch: byte-identical z. Cross-arch: deterministic under Strict (no
  transcendentals anywhere in this feature).

## IJob / warm-state compliance

Both structs follow the fProxyIC0 pattern: readonly, every piece of state behind arena-owned
native buffers, ALL fields assigned at construction, nothing scalar ever mutated afterward — an
IJob struct-copy loses nothing because there is nothing to lose (the warm-state failure mode is
post-construction mutation of struct fields; `Scratch` mutation is buffer CONTENTS through a
stable pointer, which survives copies). No `RunByRef` dependence. Mandatory test: build once,
run the full preconditioned solve INSIDE an `IJob` (`.Run()`), twice on the same instance —
per the LOBPCG cache-copy lesson (test double-buffered/cached solvers THROUGH an IJob).

## Tests

`TemplateSourceTests/fProxy/SchwarzPreconditionerTests.fProxy.cs` (templated, both scalar types):

1. **Exactness**: one subdomain covering the whole matrix (subdomainSize ≥ N), δ=0 → M = A⁻¹ →
   `pcg` converges in 1 iteration on `fProxyRandomSparseSPD`. Also `Apply` output vs dense
   `CHO.solveInPlace` on the same A: match to factor tolerance.
2. **Symmetry of AS**: random r₁, r₂; `⟨M⁻¹r₁, r₂⟩ ≈ ⟨r₁, M⁻¹r₂⟩` (relative tol) on
   `fProxyLaplacian2D` and `fProxyPenalizedGrid3D`, several (subdomainSize, δ) combos including
   scrambled-ish sizes that force ragged last subdomains.
3. **RAS ≠ AS but both solve**: `pbiCGStab` + RAS converges on `fProxyRandomSparse`
   (diagonally-dominant general square) and on the SPD cases; solution matches direct solve.
4. **Convergence / headline (iterations)**: on `fProxyLaplacian2D` (64×64, 128×128) and
   `fProxyPenalizedGrid3D` (~12³-16³, stiff penalty — the case where IC(0) needs shifts and many
   iterations), assert `iters(pcg+AS) < iters(pcg+IC0)` and `< iters(pcg+BlockJacobi)` at the
   default options, same tol. The elasticity/penalty case is the expected headline win (strong
   local solves absorb the penalty stiffness inside subdomains); Poisson at default s may need
   δ=2 or s=256 to beat IC0 — the test pins whatever (s, δ) wins and the DEVLOG records the
   sweep. If NO setting beats IC0 on Poisson, the test asserts the elasticity win only and the
   docs claim is scoped accordingly (honest-headline rule).
5. **Symmetric-storage input**: `Symmetric=true` A produces bit-identical M⁻¹r to the same A in
   full storage.
6. **Determinism**: two builds+applies → byte-identical z (xxHash32 harness); add a row to the
   determinism-conformance A-groups when that harness lands.
7. **Through-IJob**: as in the IJob section; result identical to main-thread run.
8. **Breakdown path**: an indefinite/near-singular local block → out-info reports the shift
   (AS) / Singular (RAS local LU with zero pivot column) without throwing; throwing ctor throws.

Benchmark: `Benchmarks/` templated file, Tools/benchmark.ps1 protocol — wall-clock setup and
per-iteration apply vs IC0/SSOR/BlockJacobi on both galleries, s ∈ {64,128,256} × δ ∈ {1,2},
reporting iterations AND total solve time. Gate for the MVP is the ITERATION claim (test 4);
wall-clock is reported, not gated (open question).

## Phasing

1. **MVP (this spec)**: contiguous partition, δ-layer overlap, dense CHO/LU local factors,
   AS + RAS structs, arena factories, ladder rungs, tests 1-8, benchmark.
2. **Phase 2 (optional, measured need)**: greedy-BFS connected partitioning; packed-triangular
   factor storage (halves memory); `pbiCGStab`+AS rungs if asked.
3. **Two-level**: coarse space via the multigrid spec's aggregation prolongator (D6) — separate
   spec once `spec-multigrid-solver.md` Phase 1 exists.

## File layout

- `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxySchwarz.cs` — both structs + shared
  internal setup (partition/overlap/gather/factor). One file because they share ~80% of setup;
  split later only if it grows past taste.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/SchwarzOptions.cs` — shared options struct
  (no proxy tokens; PreconditionerInfo precedent).
- Arena factories appended to `Sparse/Arena.Sparse.fProxy.cs`.
- Ladder rungs appended to `OP/Krylov.fProxy.cs`, `OP/Krylov.PMinres.fProxy.cs`,
  `OP/Krylov.PBiCGStab.fProxy.cs`.
- Tests `TemplateSourceTests/fProxy/SchwarzPreconditionerTests.fProxy.cs`; benchmark file beside
  the existing sparse-solver benchmarks. Regenerate via `Tools/regen.ps1`; DEVLOG entries in
  `TemplateSource/Sparse/DEVLOG.md` (perf sweep results, rejected-alternative notes live there,
  never in code comments).

## Open questions for the user

1. **Naming**: `fProxyAdditiveSchwarz` / `fProxyRestrictedSchwarz` (spec'd) vs shorter
   `fProxyAS` / `fProxyRAS` (IC0/ILU0-style terseness) — pick one before codegen.
2. **Default subdomainSize=128, overlap=1** — sized for the 10³-10⁵ unknown regime and the
   O(N·g²·s) factor memory. OK, or default smaller (64) to be memory-conservative?
3. **Headline gate**: iterations-only (spec'd), or must AS also beat IC0 in WALL-CLOCK on at
   least one gallery before it ships?
4. **`pbiCGStab` rungs for symmetric AS** (legal, slower than RAS): add, or keep the ladder
   minimal (spec'd: no)?
5. **Packed-triangular factor storage** (halves the dominant memory cost, slightly hairier
   sweeps): MVP or Phase 2 (spec'd: Phase 2)?
6. **Greedy-BFS partitioning**: acceptable to ship MVP with contiguous-ranges-only (degrades
   gracefully on scrambled numberings), BFS deferred to Phase 2 (spec'd: yes)?
7. **MINRES caveat**: AS is SPD whenever the build reports Success (D4), so `pminres` rungs are
   spec'd in. Drop them anyway to keep the first cut smaller?
