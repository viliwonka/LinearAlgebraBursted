# Spec: sparse approximate inverse preconditioners over BSR — FSAI + SPAI

Status: DRAFT (not implemented). Internal dev doc.
Scope: two new `IfProxyPreconditioner` implementations in `TemplateSource/Sparse/` —
`fProxyFSAI` (factored, SPD, for `pcg`/`pminres`) and `fProxySPAI` (general, for
`pbiCGStab`) — plus arena factories, Krylov overload rungs, tests, benchmarks.

## 1. Motivation

Goal: a generic preconditioner better than IC(0) *in this library's execution model*.

IC(0)'s Apply (`fProxyIC0.cs:286`) is one forward and one backward block-triangular
sweep — inherently SEQUENTIAL row-to-row dependencies. Burst cannot vectorize across
rows; each PCG iteration pays a serial latency chain over all block rows.

A sparse approximate inverse stores `M ≈ A⁻¹` (FSAI: a triangular factor `G` with
`GᵀG ≈ A⁻¹`) as an explicit sparse BSR matrix. Applying it is a plain BSR spMV —
the same block-unrolled, SIMD-friendly kernel `BSR.spMV` the operator itself uses
(`SparseOP.fProxy.cs:21`). No triangular solve at apply time, ever. Setup does more
work (many small independent dense solves), Apply does strictly stream-friendly work.
Deterministic, fixed-order, and trivially safe through IJob struct copies (set-once
readonly buffers, like IC0).

Rows/columns of the approximate inverse are computed INDEPENDENTLY of each other —
no sequential factorization dependency like IC(0)/ILU(0) — so setup has no breakdown
cascade and (post-MVP) could even be job-parallelized without changing results.

## 2. Existing precedents to match (surveyed 2026-07-18)

| Piece | Precedent | File |
|---|---|---|
| Preconditioner interface | `IfProxyPreconditioner.Apply(in fProxyN r, ref fProxyN z)`, z must not alias r | `TemplateSource/Interfaces/LinearOperator.fProxy.cs:54` |
| SPD preconditioner struct shape | `fProxyIC0`: `readonly struct`, fields `fProxyBSR L` + `fProxy Shift`, arena-composed, NO record table, NO Dispose, throwing ctor + `out PreconditionerInfo` twin | `TemplateSource/Sparse/fProxyIC0.cs` |
| Nonsymmetric sibling | `fProxyILU0` (for `pbiCGStab`), BR ≤ 16 guard precedent | `TemplateSource/Sparse/fProxyILU0.cs` |
| Arena factory | `arena.fProxyIC0(in A)` + `arena.fProxyIC0(in A, out PreconditionerInfo)` — thin `Arena self = this; return new ...(in A, ref self, ...)` | `TemplateSource/Sparse/Arena.Sparse.fProxy.cs:235` |
| Build outcome | `PreconditionerInfo { DirectSolveStatus status; double shift; int attempts; Solved }` | `TemplateSource/Sparse/PreconditionerInfo.cs` |
| pcg rung set (3 rungs per preconditioner) | zero-alloc (`ref r/p/Ap/z` scratch), arena-scratch (`maxIter, tol`), defaults (`A.M_Rows`, `Consts.fProxySqrtEps`); each forwards via `new fProxyBSROperator(in A)` into generic `pcg<TOp,TPre>` | `TemplateSource/OP/Krylov.fProxy.cs:439-473` (IC0 rungs) |
| pminres rung set | same 3 rungs for BlockJacobi/SSOR/IC0 | `TemplateSource/OP/Krylov.PMinres.fProxy.cs:230-342` |
| pbiCGStab rung set | ILU0 rungs (`:115`, `:132`) | `TemplateSource/OP/Krylov.PBiCGStab.fProxy.cs` |
| BSR alloc with known pattern | IC0 fills `arena.fProxyBSR(nb, nb, BR, BR, nnzb, uninit:true)` RowPtr/ColInd by hand (pattern known up front — no builder needed) | `fProxyIC0.cs:87-104` |
| Explicit transpose | `arena.fProxyBSRTranspose(in A)` — O(nnz) one-time | `Arena.Sparse.fProxy.cs:132` |
| Symmetric-storage mirror | `arena.fProxyBSRMirrorToFull(in A)` | `Arena.Sparse.fProxy.cs:173` |
| Small dense SPD solve | `CHO.decompInPlace(ref A_to_L)` / `CHO.decompSolve(ref L, ref b_to_x)` (also MxN RHS overload `CHO.fProxy.cs:274`) | `TemplateSource/OP/CHO.fProxy.cs` |
| Owned apply scratch | `fProxyColScaledOperator.Scratch` (owned vector, aliasing contract in doc) | `LinearOperator.fProxy.cs:158` |
| Test matrices | `arena.fProxyLaplacian2D(gridX, gridY)` (Poisson), `arena.fProxyPenalizedGrid3D(nx,ny,nz,EA,penalty)` (elasticity), `fProxyRandomSparseSPD`, `fProxyRandomSparse` | `TemplateSource/Sparse/Gallery.Sparse.fProxy.cs` |
| Test files to mirror | `SparseIC0Tests.fProxy.cs`, `SSORTests.fProxy.cs`, `SparseSolverTests.fProxy.cs` | `TemplateSourceTests/fProxy/` |
| Benchmarks to extend | `PCGBenchmark.fProxy.cs`, `LargeSparseBenchmark.fProxy.cs` | `TemplateSourceBenchmarks/` |

Determinism rules (see `docs`/memory determinism-analysis): fixed iteration order,
`+ - * / sqrt` only under FloatMode.Strict for cross-arch claims; SIMD allowed,
reassociation not. Both algorithms below satisfy this: gathers are ascending-index
scans, local solves are Cholesky (sqrt) / substitution, application is the existing
spMV kernel. No transcendentals anywhere.

Warm-state IJob lesson (LOBPCG cache-copy bug, warm-state audit): NO mutable scalar
fields, no double-buffer identity. Both structs below are `readonly`, all state is
set-once arena-tracked buffers fixed at construction — the same shape as `fProxyIC0`,
which is already IJob-copy safe. Still add a through-IJob test (section 8).

## 3. fProxyFSAI — factored sparse approximate inverse (SPD; the pcg default candidate)

### 3.1 Math

`G` block-lower-triangular over a PRESCRIBED static pattern `S`; construction makes
`G A Gᵀ ≈ I`, hence `M = Gᵀ G ≈ A⁻¹`. `M` is symmetric by construction and SPD
whenever `G` is nonsingular (triangular, positive diagonal — guaranteed by the
Cholesky scaling below), so FSAI is a VALID CG preconditioner. This is the
Kolotilina–Yeremin / Kaporin FSAI.

Per block-row `i`, with block pattern `Jᵢ = { j : (i,j) ∈ S, j ≤ i }` (diagonal
block always included; sorted ascending):

1. Gather the dense SPD submatrix `Â = A[Jᵢ, Jᵢ]` — size `(|Jᵢ|·BR) × (|Jᵢ|·BR)`.
2. Solve `Â · X = E`, where `E` is zero except an `I_BR` identity in the diagonal
   (last) block slot — i.e. `X = Â⁻¹ E`, `|Jᵢ|·BR × BR`. Via `CHO.decompInPlace`
   + the multi-RHS `CHO.decompSolve` on `Allocator.Temp` scratch.
3. Block Kaporin scaling: `D = X_last` (the `BR×BR` block of `X` at the diagonal
   slot) is SPD and symmetric; factor `D = C·Cᵀ` (dense Cholesky). The row-block of
   `G` is `g = C⁻¹ · Xᵀ` (forward-substitute `C` against each of `X`'s columns).
   Then `g Â gᵀ = C⁻¹ D C⁻ᵀ = I_BR` exactly — `diag-block(G A Gᵀ) = I`.
4. Scatter `g` into `G`'s value array at row `i`'s slots (pattern was fixed in
   advance, so this is a straight copy).

For `BR = 1` this degenerates to the classic scalar FSAI
(`gᵢ = ĝ / sqrt(ĝ_last)`, `ĝ = Â⁻¹ e`).

Rows are mutually independent: fixed ascending `i` loop, fixed ascending gather
order ⇒ bit-reproducible. No global breakdown: if `CHO` reports a non-SPD local
system, retry THAT row with a scaled diagonal shift on `Â` (escalation ladder and
`pivotFloor`/`diagMax` scale exactly as in `fProxyIC0.cs:106-135`), record the
worst shift in `Shift`/`PreconditionerInfo`. If the largest shift still fails,
build fails with `DirectSolveStatus.NotPositiveDefinite` (out-info twin) / throws.

### 3.2 Pattern

MVP default: `S` = block pattern of `lower(A)`, diagonal included — exactly the
pattern IC(0) uses, so comparisons are like-for-like and pattern extraction reuses
IC0's `col <= i` scan (`fProxyIC0.cs:70-104`). Symmetric-storage `A` already stores
exactly this pattern (consume zero-copy, same as IC0); full-storage `A` reads only
its lower blocks. Missing diagonal block ⇒ throw (contract, same message shape as
IC0). Gathering `A[j1,j2]` for `j2 > j1` under symmetric storage takes the
transpose of the stored `A[j2,j1]` — a small gather helper shared by FSAI and SPAI.

Deferred (phase 3): pattern of `lower(A²)` (denser, stronger — often halves
iterations at 2-4× setup/apply cost) behind `patternPower = 2`. The symbolic
`A²` block pattern is a deterministic sorted merge; no numerics involved.

Optional pattern filtering (`dropTol`, default 0 = off): exclude off-diagonal
block `(i,j)` from `S` when `‖A[i,j]‖_F ≤ dropTol · sqrt(‖A[i,i]‖_F · ‖A[j,j]‖_F)`.
Data-dependent but a pure function of `A` with fixed evaluation order ⇒ still
deterministic. `sqrt`-only.

### 3.3 Struct

```csharp
public readonly struct fProxyFSAI : IfProxyPreconditioner
{
    public readonly fProxyBSR G;    // block-lower-triangular factor, pattern S
    public readonly fProxyBSR Gt;   // Gᵀ, materialized once via arena.fProxyBSRTranspose
    public readonly fProxyN  Scratch; // length Rows: holds y = G·r during Apply
    public readonly fProxy Shift;   // worst per-row rescue shift; 0 = clean
    public int Rows => G.M_Rows;

    public fProxyFSAI(in fProxyBSR a, ref Arena arena);                              // throws on failure
    public fProxyFSAI(in fProxyBSR a, ref Arena arena, out PreconditionerInfo info); // non-throwing
    // (options overloads: see 5. API)

    // z = Gᵀ (G r): two forward BSR spMVs through Scratch. z must not alias r or Scratch.
    public void Apply(in fProxyN r, ref fProxyN z);
}
```

- Arena-composed like IC0: no record table of its own, no `Dispose()`.
- `Gt` explicit (one extra `fProxyBSRTranspose` at build, ~2× G's memory) so BOTH
  applies are forward `BSR.spMV` — the scatter-traversal `BSR.spMVT` exists but is
  the slow path (see `fProxyBSROperator.cs` rationale for precomputed transposes).
- `Scratch` is an owned arena vector (precedent: `fProxyColScaledOperator.Scratch`);
  Apply's aliasing guard must check `z`/`r`/`Scratch` pairwise pointer inequality.
- Shape guards identical to IC0: square (`BlockRows==BlockCols`, `BR==BC`), every
  diagonal block stored.

### 3.4 Costs

- Setup: per row one dense Cholesky solve of size `nᵢ = |Jᵢ|·BR` ⇒ `Σᵢ O(nᵢ³)`
  plus one `BR³` Cholesky. Poisson 2D (`fProxyLaplacian2D`, ≤ 3 lower blocks/row,
  BR=1): trivial. `fProxyPenalizedGrid3D` (BR=3, ≤ ~14 lower blocks/row): ~42³
  flops/row worst case — still small, and strictly O(rows) total. Compare IC0
  setup: one sequential sparse factorization; same order of magnitude.
- Apply: 2 spMVs over nnzb(G) ≈ nnzb(lower(A)) ⇒ ≈ 1 spMV over nnzb(A) worth of
  flops — comparable flop count to IC0's two sweeps but on the streaming kernel.
- Memory: `G` + `Gt` ≈ nnzb(A) + nb diagonal blocks total (vs IC0's L ≈ half that).
  Plus `Scratch` (n).

## 4. fProxySPAI — general sparse approximate inverse (nonsymmetric; for pbiCGStab)

### 4.1 Math

Explicit `M ≈ A⁻¹` minimizing `‖M A − I‖_F` over a static pattern, ROW-oriented —
chosen over the textbook column form `‖I − A M‖_F` because `fProxyBSR` is row-major:
row `i` of `M A` reads whole rows of `A`, which is the natural access. (The row form
is exactly column-SPAI on `Aᵀ`; algebraically equivalent family.)

`‖M A − I‖²_F = Σᵢ ‖mᵢ A − eᵢᵀ‖²` — each row `mᵢ` (supported on block pattern
`Jᵢ`) is an INDEPENDENT small least-squares problem:

1. `Jᵢ` = block pattern of row `i` of `A` (MVP default; diagonal block required).
2. Shadow pattern `Iᵢ` = sorted union of the block patterns of rows `j ∈ Jᵢ`
   (the block columns `mᵢ A` can touch). Deterministic merge.
3. Dense local LS: `min ‖Âᵀ m − ê‖₂` with `Â = A[Jᵢ, Iᵢ]` gathered
   (`|Jᵢ|·BR × |Iᵢ|·BR`), `ê` = `eᵢ` restricted to `Iᵢ` (an `I_BR` block in the
   diagonal slot). Solve the block variant (RHS is `BR` columns) via NORMAL
   EQUATIONS: `N = Â Âᵀ` (`|Jᵢ|·BR` square, SPD when `Â` has full row rank),
   `CHO.decompInPlace` + multi-RHS `decompSolve`. Rank-deficient local system
   (CHO breakdown): apply the same escalating diagonal-shift rescue as FSAI
   (a Tikhonov-regularized row — still a usable approximate-inverse row);
   record worst shift. Normal equations chosen over local QR for simplicity and
   because local systems are tiny and shift-rescued; revisit only if tests show
   accuracy loss (open question Q5).
4. Scatter `m` into `M`'s row `i` (pattern fixed up front).

`M` is NOT symmetric even for symmetric `A` ⇒ NOT a valid CG/MINRES
preconditioner. Contract: route to `pbiCGStab` only; XML doc on the struct states
the CG-invalidity explicitly (mirror the ILU0 "nonsymmetric sibling" phrasing).

Symmetric-storage `A` pays a one-time `fProxyBSRMirrorToFull` (same as ILU0), since
SPAI needs full rows.

### 4.2 Struct

```csharp
public readonly struct fProxySPAI : IfProxyPreconditioner
{
    public readonly fProxyBSR M;    // pattern = pattern(A) (MVP), full storage
    public readonly fProxy Shift;   // worst per-row Tikhonov rescue; 0 = clean
    public int Rows => M.M_Rows;

    public fProxySPAI(in fProxyBSR a, ref Arena arena);
    public fProxySPAI(in fProxyBSR a, ref Arena arena, out PreconditionerInfo info);

    public void Apply(in fProxyN r, ref fProxyN z);  // z = M·r: ONE BSR spMV. z must not alias r.
}
```

No scratch vector needed (single spMV). Costs: setup `Σᵢ O(|Jᵢ|²·|Iᵢ|·BR³)`
(forming `N`) + `O((|Jᵢ|BR)³)` per row; apply = one spMV over nnzb(A); memory =
nnzb(A) blocks (pattern of A) — same as ILU0's F.

## 5. API

Arena factories in `Arena.Sparse.fProxy.cs`, matching the IC0/ILU0 style
(`Arena self = this; return new ...`):

```csharp
public fProxyFSAI fProxyFSAI(in fProxyBSR A);
public fProxyFSAI fProxyFSAI(in fProxyBSR A, out PreconditionerInfo info);
public fProxyFSAI fProxyFSAI(in fProxyBSR A, in SaiOptions opts);
public fProxyFSAI fProxyFSAI(in fProxyBSR A, in SaiOptions opts, out PreconditionerInfo info);
// same four for fProxySPAI
```

Options — a SHARED non-proxy struct (new file `TemplateSource/Sparse/SaiOptions.cs`;
codegen only reads `*.cs`, and non-proxy shared types are the established route,
cf. `PreconditionerInfo`, `Pivot`/`Indices`). Fields are `double`/`int` so no
proxy token and no CS1750 default-param trap ([[template-default-params-cs1750]]):

```csharp
public struct SaiOptions
{
    public int patternPower;   // 1 = pattern(A) [MVP default]; 2 = pattern(A²) [phase 3, throw until then]
    public double dropTol;     // 0 = keep full pattern [default]; else block-norm filter (3.2)
    public static SaiOptions Default => new SaiOptions { patternPower = 1, dropTol = 0 };
}
```

The no-options overloads forward `SaiOptions.Default`.

Krylov rungs — mirror the existing ladders EXACTLY (three rungs, forwarding via
`fProxyBSROperator` into the generic core; doc comments follow the IC0 rungs'
wording):

- `Krylov.fProxy.cs`: 3 × `pcg(in fProxyBSR A, in fProxyFSAI M, ...)` after the
  IC0 rungs (`:445-473`).
- `Krylov.PMinres.fProxy.cs`: 3 × `pminres(..., in fProxyFSAI M, ...)` after the
  IC0 rungs. Caveat in the doc: FSAI construction needs the local `A[J,J]` SPD;
  on an indefinite A it may fall back to shifted rows (same practical caveat IC0
  already has on pminres).
- `Krylov.PBiCGStab.fProxy.cs`: `pbiCGStab(..., in fProxySPAI M, ...)` rungs
  mirroring the ILU0 rungs (`:115`, `:132`) one-for-one.
- NOT in MVP: a `lobpcg(..., in fProxyFSAI M, ...)` convenience rung (BlockJacobi
  has one; FSAI is SPD-valid there too). Trivial follow-up; keep MVP surface small.

## 6. Determinism

- Pattern: pure function of `A`'s pattern (+ optional dropTol block norms), built
  with fixed ascending scans/merges. No data-dependent growth (that is exactly
  what adaptive SPAI adds — deferred, section 9).
- Local solves: gathers in ascending index order; `CHO`/forward-substitution =
  `+ - * / sqrt` only. Fixed row loop `i = 0..nb-1`. Shift ladder is a fixed
  deterministic escalation (IC0 precedent).
- Apply: the existing `BSR.spMV` kernel — already used by every BSR solve path and
  covered by the library's determinism posture (fixed-order block-unrolled sums).
- Struct: `readonly`, set-once arena buffers, no mutable fields ⇒ value copies
  into IJobs are exact and stateless (no warm-state hazard by construction).

## 7. File layout

```
TemplateSource/Sparse/fProxyFSAI.cs          new — struct + build + Apply
TemplateSource/Sparse/fProxySPAI.cs          new — struct + build + Apply
TemplateSource/Sparse/SaiOptions.cs          new — shared non-proxy options
TemplateSource/Sparse/Arena.Sparse.fProxy.cs edit — 8 factory overloads
TemplateSource/OP/Krylov.fProxy.cs           edit — 3 pcg rungs (FSAI)
TemplateSource/OP/Krylov.PMinres.fProxy.cs   edit — 3 pminres rungs (FSAI)
TemplateSource/OP/Krylov.PBiCGStab.fProxy.cs edit — pbiCGStab rungs (SPAI)
TemplateSourceTests/fProxy/SparseFSAITests.fProxy.cs   new (mirror SparseIC0Tests)
TemplateSourceTests/fProxy/SparseSPAITests.fProxy.cs   new
TemplateSourceBenchmarks/PCGBenchmark.fProxy.cs        edit — add FSAI arm
TemplateSource/Sparse/DEVLOG.md              edit — measured numbers, decisions
```

Regenerate via `Tools/regen.ps1`; tests headless via `Tools/*.ps1`. Shared
gather helper (`A[j1,j2]` under symmetric/full storage) lives in the FSAI file or
`UnsafeOP.Sparse` — implementer's choice, but FSAI and SPAI must share it.

## 8. Tests and benchmarks

Correctness (per proxy type, templated):
1. Exactness: diagonal-blocks-only SPD A ⇒ FSAI `M = A⁻¹` exactly; `pcg` converges
   in 1 iteration. Tiny dense-lower-pattern A (pattern = full lower triangle) ⇒
   `G = chol(A)⁻¹` up to roundoff, `pcg` in ≤ 2 iterations.
2. SPD preservation (FSAI): random SPD BSR (`fProxyRandomSparseSPD`); check
   `dot(r1, M r2) == dot(r2, M r1)` to tight tolerance and `dot(r, M r) > 0` for
   several random r — the CG-validity contract.
3. Convergence: `fProxyLaplacian2D` and `fProxyPenalizedGrid3D` — FSAI-`pcg`
   reaches tol; iterations strictly fewer than block-Jacobi-`pcg` (hard assert),
   and recorded vs IC0-`pcg` (soft: log; expectation is 1.0-1.5× IC0's count with
   the same pattern — the win is apply shape, not iteration count; see headline
   benchmark below).
4. SPAI residual quality: `‖M A − I‖_F < ‖D⁻¹ A − I‖_F` (beats Jacobi) on a
   nonsymmetric diagonally-dominant `fProxyRandomSparse`-derived square BSR;
   `pbiCGStab`+SPAI converges where the existing ILU0 test matrix converges.
5. Non-throwing twins: engineered breakdown (indefinite A for FSAI) ⇒
   `info.Solved == false` / correct status, no throw; throwing ctor throws.
6. Guards: non-square, missing diagonal block, aliasing (`z==r`, and `z==Scratch`
   for FSAI) throw.
7. Through-IJob determinism ([[lobpcg-burst-eigenvector-bug]] lesson): build once
   on the main thread, run `pcg`(FSAI) inside a Burst `IJob.Run()` AND on the
   managed path; assert identical iteration counts and bit-identical x between two
   repeated IJob runs. Same for SPAI+`pbiCGStab`.
8. Symmetric-storage vs full-storage A produce bit-identical G (FSAI).

Benchmark (headline: "beats IC(0)"): extend `PCGBenchmark.fProxy.cs` with an FSAI
arm next to the existing BlockJacobi/SSOR/IC0 arms, on `fProxyLaplacian2D` (large
grid) and `fProxyPenalizedGrid3D`. Record per arm: setup ms, iterations, total
solve ms, ms/iteration. Acceptance: FSAI total solve wall-clock ≤ IC0's on at
least the large Poisson case (apply-throughput win must show up end-to-end);
iterations < block-Jacobi everywhere. Numbers go in `Sparse/DEVLOG.md`, not
comments (comment policy).

## 9. Phasing

- Phase 1 (MVP): `fProxyFSAI`, pattern(A), dropTol, arena factories, pcg+pminres
  rungs, tests 1-3/5-8, benchmark arm. Ships alone if phase 2 slips.
- Phase 2: `fProxySPAI` + pbiCGStab rungs + tests 4-7.
- Phase 3 (deferred, separate spec if wanted): `patternPower = 2` (A² pattern) for
  FSAI; ADAPTIVE SPAI (Grote–Huckle dynamic pattern growth by largest residual
  entries) — stronger, but pattern selection is data-dependent: determinism
  requires pinning the growth rule (fixed candidate ordering, fixed ties, fixed
  growth counts). Explicitly out of MVP. Also: lobpcg FSAI rung; optional
  job-parallel setup (rows independent — results unchanged by construction).

## 10. Open questions

- Q1 Default FSAI pattern: lower(A) (cheap, like-for-like with IC0) vs lower(A²)
  (stronger). MVP answer: lower(A); revisit after benchmark numbers exist.
- Q2 dropTol default 0 (keep full pattern) — expose but leave off?
- Q3 Ship order: FSAI-only first (phase 1 alone) or FSAI+SPAI together? SPAI has
  no in-repo baseline pressure (ILU0 already covers pbiCGStab); FSAI is the
  motivated piece.
- Q4 Store `Gt` explicitly (2× memory, fast apply — spec'd default) vs on-the-fly
  `BSR.spMVT` (half memory, scatter-slow)? Could be a `SaiOptions` flag if memory
  ever matters.
- Q5 SPAI local solve: normal equations + CHO (spec'd) vs small QR — switch only
  if test 4 shows conditioning problems on the local systems.
- Q6 Does FSAI get the pminres rungs in MVP at all, given the SPD-local-solve
  caveat on indefinite A? (Spec'd yes, with doc caveat, mirroring IC0's presence
  there.)
