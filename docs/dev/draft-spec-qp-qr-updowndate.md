# QP v2: factorization up/downdating in the active-set loop (draft spec)

Status: STAGE 1 + STAGE 2 SHIPPED 2026-07-14. Stage 1 = hybrid-store op-log QR (option (a)).
Stage 2 = explicit persistent Z/QZ/H_Z up/downdate (option (b); GGMS/Goldfarb-Idnani Cholesky
update NOT done — an add's congruence re-triangularization is O(nz³)). Measured GO on a
loop-isolating QPBenchmark section: incr vs batch −11..−33% (float) / −12..−48% (double) at
n=16/64/192, growing with n; full facade at n=192 ≈ half the stage-1 baseline. COLD core defaults
incremental, WARM/MPC core defaults batch for a SINGLE solve. Stage 2b (also shipped 2026-07-14) adds
CROSS-TICK persistence for MPC: fProxyMPCState carries the factorization + reduced space across ticks
and reuses them wholesale when the working set is unchanged (qpActiveSetCoreWarmPersistent) — warm
steady-state per-frame cost −25..−34% (float) / −46..−55% (double) box, −58..−74% with active general
rows. See `TemplateSource/OP/DEVLOG.md` under "QP" (incl. the IJob-copy metadata trap and the future
incremental-diff-repair extension). Target: `TemplateSource/OP/QP.fProxy.cs`.

## Problem

`qpActiveSetLoop` re-does, on EVERY iteration (one constraint add or drop per
iteration):

1. `AssembleWorkingSetTranspose` — rebuild A_Wᵀ (n×k) from wstatus.
2. `FactorWorkingSetTranspose` — Householder QR of A_Wᵀ, O(nk²).
3. `FormNullSpaceBasis` — Z (n×nz) by reverse reflector sweep, O(k·n·nz).
4. `SolveReducedNewtonStep` — QZ = Q·Z, H_Z = ZᵀQZ (O(n²nz), dominant), Cholesky
   of H_Z (O(nz³/3)), solve.

The working set changes by exactly ONE row between consecutive iterations
(add or drop), and consecutive MPC ticks differ by a handful of rows
(`qpActiveSetCoreWarm` already counts `workingSetChanges`). All four steps are
structurally rank-one updates.

## Stage 1 — persist + up/downdate the QR of A_Wᵀ

State that survives across iterations (new struct, lives in the loop frame; a
later stage can move it into `fProxyQPState` for cross-call MPC reuse):

- AWT reflector store (n×k_max) + R (k_max×k_max) + the column→unified-row map
  (`rowOfCol` today).

Operations:
- **Add constraint** (row t enters W): append its column a_t to A_Wᵀ. Update =
  apply the k stashed reflectors to a_t (one `ApplyWorkingSetQtForward` pass,
  O(nk)), then ONE new Householder on the tail — exactly `decompInPlace`'s next
  step. R gains column k. O(nk) total vs O(nk²) refactor.
- **Drop constraint** (column j leaves): remove column j of A_Wᵀ. This is the
  QRCP downdating machinery's case (`FactorWorkingSetTranspose`'s own comment
  names it as the donor): columns j+1..k-1 shift left; R becomes upper-Hessenberg
  in columns j..k-2; restore triangularity with k-1-j Givens rotations applied to
  R's rows (O(k²)) and ACCUMULATE the rotations... note the Householder store
  cannot absorb Givens rotations directly — see "representation decision" below.

**Representation decision (the crux):** the current implicit-Q (stashed
Householder vectors) representation cannot cheaply absorb the Givens rotations a
column-drop produces. Options, in recommended order:
  a. **Hybrid store**: keep the Householder vectors AND a compact list of Givens
     rotations appended by drops; `ApplyWorkingSetQtForward` replays reflectors
     then rotations (rotations act on the transformed k-space only, O(#rot) per
     apply). Drops are cheap; the rotation list resets whenever a full refactor
     happens. Bounded growth: refactor from scratch when #rotations > c·k (the
     amortized cost stays O(nk) per change).
  b. Materialize Q1/Z explicitly (n×n storage, updates via plain Givens on
     columns) — simpler math, more memory traffic; the FormNullSpaceBasis cost
     disappears entirely into incremental column maintenance.
  Decide by benchmark on MPC-shaped problems (n≈20-60, k changes 1-4/tick).

## Stage 2 — persist + incrementally maintain the reduced space (Z, QZ, H_Z)

Status: DESIGNED + implementation-ready (this section). RECOMMENDATION: **GO**,
option **(b)** — maintain `Z`, `QZ = Q·Z`, and `H_Z = ZᵀQZ` EXPLICITLY as
persistent dense buffers, up/downdated in O(n·nz) per active-set change, and
recompute `chol(H_Z)` FROM SCRATCH each iteration (do NOT attempt a
Gill-Golub-Murray-Saunders Cholesky up/downdate — see "why not (a)/(c)"). The win
comes from eliminating the two O(n²·nz) per-iteration terms (`FormNullSpaceBasis`
+ fresh `QZ = Q·Z`), not from the Cholesky.

### What dominates today (per iteration of `qpActiveSetLoop`)

`FormNullSpaceBasisFromFactor` (O(#ops·n·nz)) and `SolveReducedNewtonStep`'s
`QZ = Q·Z` (O(n²·nz)) are each ~⅜ of the per-iteration flops and together ~¾ of
it. `H_Z = ZᵀQZ` (O(n·nz²)) is next; the Cholesky (O(nz³/3)) and the fixed
O(n²)/O(mn) terms (`Qx`, `Qp`, `ApplyFactorQtForward`, `A·x`, `A·p`) are the rest.
Stage 2 attacks the first three; the fixed terms are out of scope.

### The update algebra (the crux the deferral hinged on)

The op-log already gives everything needed; the reduced buffers ride ALONGSIDE the
Stage-1 log (which is unchanged). Keep `Z`/`QZ` in the SAME column frame the log's
`FormNullSpaceBasisFromFactor` produces (log-trailing-coordinate order) so the
Householder reflector a add appends aligns index-for-index with `Z`'s columns for
free, and so the buffers can be validated column-for-column against a fresh
`FormNullSpaceBasisFromFactor`.

**DROP** (column j leaves; k→k−1, nz→nz+1). `DropFromFactor` appends k−1−j Givens
rotations `G` acting only on coords j..k−1; `Q̂_new = Q̂_old·Gᵀ`. `Gᵀ` mixes only
coords j..k−1, so columns k..n−1 of `Q̂` are UNCHANGED: the old nz columns of `Z`
survive verbatim, and exactly ONE new orthonormal column enters at coord k−1.
This is classical bordering, confirmed against the op-log:
- `z_new = Q̂_new·e_{s.k}` (reverse-sweep the FULL log over the single seed `e_{s.k}`,
  O(#ops·n)) — a new helper `FormNullSpaceColumn(s, seedRow, ref col)`, the
  one-column form of `FormNullSpaceBasisFromFactor`.
- Prepend: `Z_new = [z_new | Z_old]`, `QZ_new = [q_new | QZ_old]` with
  `q_new = Q·z_new` (ONE GEMV, O(n²) — the only irreducible cost of a drop).
- Border `H_Z`: `H_Z_new = [[α, wᵀ],[w, H_Z_old]]`, `α = z_newᵀ·q_new` (O(n)),
  `w_i = z_newᵀ·QZ_old[:,i] = dot(z_new, QZ_old col i)` (O(n·nz)).
- Physical prepend = shift `Z`/`QZ` columns right by one, `H_Z` down/right by one,
  then write column/row 0. O(n·nz).

**ADD** (row t enters; k→k+1, nz→nz−1). `TryAddToFactor` appends one Householder
`H_new = I − u·uᵀ`, u supported on rows [k,n), so `Q̂_new = Q̂_old·H_new` and
`H_new` mixes only coords k..n−1. Restricted to the old null-space frame this is a
size-nz reflection `Ĥ = I_nz − û·ûᵀ`, `û = u[k:n]` (the reflector tail, read for
FREE from `s.V[k:.., col]`, `col = s.reflCount−1`; the library convention gives
`ûᵀû = 2`, so `Ĥ` is orthogonal + symmetric). Then, in 5 lines:
```
Z_old·Ĥ  = Z_old − (Z_old·û)·ûᵀ         # rank-1, O(n·nz); (Z_old·û) is one n-vec
QZ_old·Ĥ = QZ_old − (QZ_old·û)·ûᵀ        # SAME right-mult; Q is NEVER re-applied
Ĥ·H_Z·Ĥ  = H_Z − û·rᵀ − r·ûᵀ,  r = p − ½β·û,  p = H_Z·û,  β = ûᵀ·p   # sym rank-2, O(nz²)
Z_new = (Z_old·Ĥ) drop LOCAL column 0;  QZ_new likewise;  H_Z_new = (Ĥ H_Z Ĥ) drop row/col 0
```
The leaving direction is exactly local column 0 (= `Q̂_new[:,k]`, the added
constraint's normal component inside null(A_W_old)); columns 1.. are the new
orthonormal null-space frame. **`Q` is never multiplied on an add** — `QZ`
transforms by the same `Ĥ` because `Q·(Z_old·Ĥ) = (Q·Z_old)·Ĥ = QZ_old·Ĥ`. Total
add cost O(n·nz), no O(n²·nz) anywhere. Delete-col-0 = shift left by one, O(n·nz).

**Cholesky**: recompute `chol(H_Z)` from scratch each iteration (copy cached `H_Z`,
factor; O(nz³/3)). This sidesteps the GGMS subtlety entirely and keeps the
regularized-retry and refactor paths trivially correct.

### Why not (a) Givens-chain Cholesky update or (c) hybrid

The add is an orthogonal CONGRUENCE by a dense size-nz Householder `Ĥ`, then a
row/col delete. Re-triangularizing `Ĥ·L` (or applying `Ĥ` as nz−1 Givens and
re-triangularizing) is O(nz³) — no cheaper than the O(nz³/3) from-scratch Cholesky
it would replace. Only a Goldfarb-Idnani-style `J = Q·R⁻ᵀ` representation makes the
added constraint fall on the LAST reduced coordinate so the update is a cheap
symmetric deletion — but that is a DIFFERENT data structure than Stage 1's op-log
QR of A_Wᵀ, i.e. a bigger rewrite. A hybrid (append-border Cholesky on drops,
which IS a genuine O(nz²) < O(nz³/3) update, but from-scratch on adds) saves only
~½ of the already-small residual Cholesky term while forcing the L-buffer to track
the prepend/rotate frame — not worth the complexity for Stage 2. Revisit
Goldfarb-Idnani as a possible Stage 3 only if the residual O(nz³/3) Cholesky
measurably dominates after Stage 2. Literature: Gill, Golub, Murray & Saunders
(1974); Goldfarb & Idnani (1983); Golub & Van Loan §6.5; Nocedal & Wright §16.5.

### Cost model (leading-term flops, mid-loop point k = nz = n/2)

| term (per iter)            | today n=64 | S2 n=64 | today n=192 | S2 n=192 |
|----------------------------|-----------:|--------:|------------:|---------:|
| FormNullSpaceBasis 2·n·nz·k |    131 072 |       0 |   3 538 944 |        0 |
| QZ = Q·Z  (n²·nz)          |    131 072 |       0 |   3 538 944 |        0 |
| H_Z = ZᵀQZ (n·nz²)         |     65 536 |       0 |   1 769 472 |        0 |
| reduced up/downdate (O(n·nz))|         0 |  ~8 200 |           0 |  ~74 000 |
| chol(H_Z) from scratch nz³/3|     10 923 |  ~9 900 |     294 912 | ~286 000 |
| fixed: Qx,Qp,ApplyQt,Ax,Ap,Zy|    20 480 |  20 480 |     202 752 |  202 752 |
| **per-iter total**         |  **~359 K**| **~44 K**|  **~9.34 M**| **~0.63 M**|
| per-iter speedup           |            | **~8×** |             | **~15×** |

Drop-iteration cost is comparable (the one q_new GEMV O(n²) ≈ 4 K/37 K replaces the
congruence). The Cholesky O(nz³/3) is the dominant residual at large n (~46% of the
n=192 stage-2 total); the fixed O(n²)/O(mn) terms are the rest.

### Expected end-to-end win on QPBenchmark — GO

Section 1 times the FULL `QP.solve` = phase-1 DualSimplex LP + active-set loop.
Only the loop is sped up, so the win is diluted by the phase-1 fraction (1−f).
At n=192 an active-set iteration (~9.3 M flops) is ~500× heavier than an LP pivot
(~O(mn) ≈ 18 K), so f (loop fraction) grows with n and is large at n≥64. With the
per-iteration loop speedup S from the table:

| n   | loop speedup S | assumed f | end-to-end total | wall-time reduction |
|-----|---------------:|----------:|-----------------:|--------------------:|
| 64  | ~8×            | ~0.5      | 0.5/8 + 0.5 ≈ 0.56 | **~44%** (1.8×)   |
| 192 | ~15×           | ~0.7–0.8  | ≈ 0.25–0.31        | **~69–75%** (3.2–4×)|

n=16 is marginal (~20%, small problem, LP-dominated) but irrelevant to the claim.
**GO**: the model puts the end-to-end reduction comfortably >10% at n≥64, and
Stage-1's own DEVLOG already measured `QZ = Q·Z` as the dominant per-iteration
cost. To de-risk the phase-1 dilution assumption, this stage ALSO adds a
loop-isolating benchmark section (see below); if A/B on THAT section shows <10%,
NO-GO — but the FLOP model says it won't for n≥64.

### State, allocation, and the fallback seam

New sibling struct `fProxyQPReducedState` (Create/Dispose paired, like
`fProxyQPFactorState`; created alongside `wsf`, disposed with it), holding:
- `Z`, `QZ` — n×n at capacity (columns 0..nz−1 live); `H_Z` — n×n (nz×nz live);
  `cholScratch` n×n and vector scratch (û·products, z_new, w) length n.
- `reducedStale` (bool), `reducedChangeCount` (int), and `const int RebuildCap`
  (≈ 16) on the struct (proxy-free-signature members MUST live on the dtype-named
  struct, never the shared partial `QP` class — same trap Stage 1 hit with
  `DeadCap`/`Create`).

No per-iteration Temp churn — all reduced buffers are allocated once per solve.

`RebuildReduced(Q, s, red)`: from-scratch `FormNullSpaceBasisFromFactor` → `Z`,
`QZ = Q·Z` (Blas.dot), `H_Z = ZᵀQZ` (Blas.dotSym) into the persistent buffers, and
clear `reducedStale`/`reducedChangeCount`. This is literally today's per-iteration
code writing into persistent storage — it IS the fallback path and the
rebuild-on-stale path.

Invalidation / drift control:
- `RefactorWorkingSet` (every DeadCap=8 drops) rebuilds the log and RE-ORDERS
  columns → set `reducedStale = true` (the next reduced solve rebuilds).
- Incremental updates accumulate rounding → also rebuild when
  `reducedChangeCount ≥ RebuildCap` (adds+drops both count; the log's DeadCap only
  counts drops, so a separate counter is needed). Amortized cost = one from-scratch
  reduced build per RebuildCap changes ≈ one old iteration / RebuildCap → negligible.
- First reduced solve of a run: `reducedStale = true` initially.

Fallback flag `useIncrementalReduced` (bool param threaded into `qpActiveSetLoop`;
`qpActiveSetCore`/`qpActiveSetCoreWarm` pass `true`, tests can pass `false`). When
`false`, skip ALL reduced maintenance and run today's from-scratch
`FormNullSpaceBasisFromFactor` + `SolveReducedNewtonStep(Q, Z, QZ, Hz, …)` every
iteration — the exact current behavior, for A/B timing and correctness diffing.

Regularized retry: `SolveReducedNewtonStep` gets a cached-`H_Z` overload
(copy `H_Z` → scratch, factor; on breakdown copy AGAIN + δ·‖Q‖∞·I, factor). Cached
`H_Z` is never destroyed (Cholesky always runs on `cholScratch`), so the retry's
"clean rebuild before adding δ" is automatic. The old Z/QZ-recomputing overload
stays for the `useIncrementalReduced == false` path.

### Where the updates hook in `qpActiveSetLoop`

- ADD accept: the SINGLE site where `TryAddToFactor` returns true and `wstatus` is
  committed (line ~1058) → `UpdateReducedOnAdd(Q, s, red)` reading `û` from the
  just-appended reflector. A rejected/dependent add (rank guard) does NOT touch the
  reduced buffers. In the guarded-retry loop, only the finally-accepted add updates.
- DROP: after `DropFromFactor` (line ~1098) → `UpdateReducedOnDrop(Q, s, red)`
  (needs `q_new = Q·z_new`). If that same drop triggers `RefactorWorkingSet`, skip
  the incremental drop-update and just set `reducedStale` (the refactor reorders
  the frame anyway).
- Edges: `nz == 0` (k == n) → no reduced space, skip (existing `haveNullSpace`
  guard). `nz` growing to n (k → 0, the empty working set the CornerToInterior test
  passes through) → `RebuildReduced`/updates must handle k = 0 (Z is the full log
  applied to I_n); verified by the k→0 test below.

### Test plan — extend `QPFactorStateTests`

Keep `UpDownDate`/`DependentAdd`/`CornerToInterior` (they pin the LOG, unchanged).
Add, in the same Burst-job Fail[] style:
- **T1 IncrementalReducedAgreement**: scripted add/drop/re-add sequence (reuse
  `UpDownDate`'s matrix). After EACH change, assert the incremental `Z`/`QZ`/`H_Z`
  match a freshly-computed `FormNullSpaceBasisFromFactor` / `Q·Z` / `ZᵀQZ`
  column-for-column to tol (the log-aligned-frame claim), PLUS
  `‖ZᵀZ − I‖`, `‖A_W·Z‖` (reuse `VerifyState`).
- **T2 across-refactor**: a drop run exceeding `DeadCap` (forces
  `RefactorWorkingSet` + `reducedStale` rebuild); assert post-rebuild agreement.
- **T3 rebuild-cap drift**: a long alternating add/drop sequence (≥ `RebuildCap`+
  changes); assert agreement stays ≤ tol across the auto-rebuild boundary.
- **T4 regularized retry**: PSD-not-PD `Q` with a null direction inside span(Z)
  forcing the δI retry; assert a valid descent `y` AND cached `H_Z` unchanged after
  the solve (`max|H_Z − fresh| ≤ tol`).
- **T5 fallback equivalence (the correctness gate)**: run `qpActiveSetLoop` with
  `useIncrementalReduced` both ways on HS21/35/52/76 + a couple brute-force
  instances; assert objective / KKT residuals agree to tol (VALUES, not iteration
  paths — per the pre-release ruling).
- **T6 k→0 edge**: CornerToInterior-shaped trajectory; assert reduced-buffer
  agreement at an interior point where nz = n (k = 0).

### Benchmark plan

- Extend `QPBenchmark` with a **loop-isolating** section: build a feasible x0 and
  call `qpActiveSetCore` directly (no phase-1 LP) at n ∈ {16,64,192}, reporting
  per-solve time — this measures the loop win WITHOUT phase-1 dilution and is the
  A/B GO/NO-GO gate (run with `useIncrementalReduced` both ways).
- Add an **MPC warm-chain** section (`qpActiveSetCoreWarm` over ~200 ticks, 1–4
  changes/tick, n_decision up to 240) — Stage 2's best case (no per-tick phase-1,
  reduced LA is nearly the whole per-tick cost).
- Section 1 (full facade) stays; record iterations (must be statistically
  unchanged — acceptance is values, not paths).

### Staged implementation order

1. `fProxyQPReducedState` (Create/Dispose, buffers, counters, `RebuildCap`).
2. `RebuildReduced` (from-scratch into persistent buffers) + wire stale/first/cap.
3. `FormNullSpaceColumn` helper; `UpdateReducedOnDrop` (border) + hook at the drop
   site; `UpdateReducedOnAdd` (congruence + delete) + hook at the add-accept site.
4. Cached-`H_Z` overload of `SolveReducedNewtonStep` (copy+factor; retry = copy+δI).
5. `useIncrementalReduced` flag threaded through `qpActiveSetLoop`; off = today's
   path verbatim.
6. Tests T1–T6; then A/B the loop-isolating benchmark. Keep incremental as default
   only if that section's win is >10% (the model says ~8×/~15× at n=64/192 → GO).

### Effort

One focused session (reduced-state struct + two update kernels + one rebuild +
solve overload + flag), plus tests. The two O(n²·nz) eliminations are the whole
prize; do NOT gold-plate with a GGMS/Goldfarb-Idnani Cholesky update (Stage 3 at
most, evidence-gated on the residual Cholesky term).

## Contracts and traps

- Bit-identity old-vs-new is NOT required (pre-release ruling), but the loop's
  DECISIONS (which constraint enters/leaves, ratio-test ties) will shift with
  rounding differences — tests must assert optimality conditions (KKT residuals,
  objective vs oracles), not iteration paths. HS21/35/52/76 oracles stay exact.
- Anti-cycling perturbation path (`usePerturbation`) rebuilds bounds, not the
  factorization — up/downdating composes with it unchanged, but the FINAL exact
  cleanup Newton step must use a FRESH factorization (it already refactors).
- Rank guard: `FactorWorkingSetTranspose` has a zeroThreshold dependent-column
  rejection. Updates must preserve it: on add, if the new column's post-transform
  tail norm < threshold, REJECT the add the same way (the loop already handles
  the rejection path).
- k can hit n (nz=0 branch) and the empty-set k=0 is currently unsupported —
  keep both behaviors.
- Allocation policy: today every iteration Temp-allocates everything. The
  persistent state must come from a caller workspace (`ref` struct param) or the
  loop's own frame; NO per-iteration Temp churn for the persistent pieces
  (matches fProxyQPState/warm-start conventions).

## Acceptance

- Full suite green; QP oracle tests unchanged (values, not paths).
- New tests: add/drop sequences that force ≥1 full-refactor fallback; rejected
  dependent add via update path; drop at j=0 / j=k-1; k→n and back.
- Benchmark: QPBenchmark n∈{16,64,192} + an MPC warm chain — expect the win to
  show as per-iteration time at n=192 and per-tick time in the MPC chain.
  Record iterations too (must be statistically unchanged).

## Effort estimate

Stage 1: one focused session (loop restructure + hybrid store + tests).
Stage 2: a second session, only if profiling justifies it.
