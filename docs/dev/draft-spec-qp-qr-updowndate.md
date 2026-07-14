# QP v2: factorization up/downdating in the active-set loop (draft spec)

Status: APPROVED direction (maintainer 2026-07-14: "I love the sound of QP v2 QR
up/downdate, yes"), not yet started. Target: `TemplateSource/OP/QP.fProxy.cs`.

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

## Stage 2 — reduced-Hessian Cholesky border/downdate (optional, measure first)

H_Z = ZᵀQZ. On drop (nz grows by 1): Z gains column z_new → H_Z gains a bordered
row/col (needs QZ column q_new = Q z_new, O(n²) — unavoidable — then the border
solve O(nz²)). On add (nz shrinks): remove the corresponding row/col — Cholesky
downdate O(nz²) — BUT the removed direction is a rotation of Z's basis, not a
trailing column, so the basis must be kept in the rotated frame consistently
(this is where Gill-Murray-Saunders gets subtle). Only build stage 2 if stage-1
profiling shows H_Z formation dominating (expected true: it is O(n²nz) today).
Literature: Gill, Golub, Murray, Saunders (1974) "Methods for modifying matrix
factorizations"; Nocedal & Wright §16.5.

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
