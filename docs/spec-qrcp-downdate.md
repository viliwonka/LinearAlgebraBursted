# Spec: QRCP norm downdating (LAPACK xGEQP3-style, guarded)

Status: APPROVED 2026-07-06. Safety-critical numerics change — the adversarial test
battery below is the acceptance gate, not an afterthought.

## Motivation

QRCP recomputes every trailing partial column norm exactly after each reflector —
O(m·(n−d)) per step, ~half of QRCP's total runtime (2048×512: QRCP.solveInPlace 72.7ms
vs QR.solveInPlace 34.5ms). Householder reflectors preserve column norms, so the norm
over rows d+1.. differs from the previous one only by the lost row-d entry:
‖col‖²(d+1..) = ‖col‖²(d..) − A[d,j]². Downdating replaces the O(m) re-sum per column
with O(1), safely — IF the catastrophic-cancellation case (remaining norm collapsed to
noise, exactly the near-rank-deficient inputs QRCP exists for) is guarded.

## Algorithm (transcribe LAPACK dgeqp3/dlaqps norm handling, unsquared)

Per column j, track TWO values: `vn1[j]` = current estimated partial norm,
`vn2[j]` = norm at the last exact computation. After step d, for each trailing j:

```
temp  = |A[d,j]| / vn1[j]                       // fraction of norm consumed by row d
temp  = max(0, (1+temp)*(1−temp))               // = 1 − temp², clamped
temp2 = temp * (vn1[j]/vn2[j])²                 // total decay since last exact norm
if (temp2 <= tol3z)                             // tol3z = sqrt(eps_machine)
    vn1[j] = exact partial norm (rows d+1..)    // vectorized re-sum (existing kernel)
    vn2[j] = vn1[j]
else
    vn1[j] *= sqrt(temp)
```

Notes:
- UNSQUARED norms (vn1/vn2 hold norms, not squares) — avoids squared-range overflow.
- Initial vn1=vn2 = exact column norms (existing vectorized sweep, once).
- `tol3z = sqrt(fProxyEpsilon)` — type-correct via proxy (float ~3.4e-4, double ~1.5e-8).
- vn1[pivot column] handling on swap: swap vn1/vn2 entries along with the columns.
- The guard MUST use vn2 (decay since last exact), not just the per-step ratio —
  gradual decay across many steps is the killer case, per LAPACK's design.
- ONE shared core: QRCP.decomp / decompInPlace / solveInPlace all route through the
  same pivoted kernel; the change lands once.

## Scratch storage (structural change)

Downdating needs persistent per-column state (vn1, vn2 — 2×n). Exact recompute needed
none, so current signatures have no home for it.

- New `fProxyQRCPCache { vn1, vn2 }` (n-sized) + Require + Arena factory, house pattern.
  (Revisits OQ-7 of the rework spec: QRCP now EARNS a cache; no dead fields.)
- New cache overloads: `decomp(..., ref cache)`, `decompInPlace(..., ref cache)`,
  `solveInPlace(..., ref cache)` — zero-alloc.
- Existing overloads keep signatures; they Temp-allocate the two n-vectors internally
  (2n « mn — validate-before-alloc per the commit-2.5 rule). ONE code path — the old
  exact-recompute loop is REMOVED from the hot path (survives only as the per-column
  guard-triggered re-sum and the init sweep).

## Adversarial test battery (the acceptance gate)

Comparison tiers — IMPORTANT, this is where naive testing goes wrong:
- **Tier E (exact-match)**: on inputs with WELL-SEPARATED trailing norms at every step,
  the pivot sequence is forced — downdated QRCP must produce the IDENTICAL Pivot and
  (up to roundoff of the norm bookkeeping — see OQ-D1) the same factors as a test-side
  reference. Any pivot deviation on separated norms = bug.
- **Tier P (property)**: on inputs with ties/near-ties or heavy cancellation, different
  but equally valid pivot orders are legitimate. Assert the INVARIANTS instead: |R| diagonal
  non-increasing; A·P == Q·R (reconstruction tol); Q orthonormal; detected rank equals
  the known rank; rank-deficient solveInPlace residual matches the exact-oracle solve's
  residual quality (not its bits).

Reference oracle: a managed (non-Burst, test-assembly-only) naive QRCP with exact
per-step norm recomputation — slow is fine, n ≤ ~64 for Tier E cases. Do NOT keep a
second Burst production path as oracle (two live paths = divergence risk).

Required cases (each float AND double; sizes both square and tall):
1. **Kahan matrix** K(n,θ) = diag(1,s,s²,…)·(I − c·upper-ones), s²+c²=1 — THE classic
   pivoted-QR stress input (graded, engineered near-rank-deficiency). Assert Tier P +
   detected rank matches SVD-based numerical rank. n = 16, 32, 64; θ near the classic
   0.285π flavor plus a sweep.
2. **Norm-collapse ladder**: A = [B | B·x + ε·noise] with ε ∈ {1e-2 … 1e-7 (float),
   1e-2 … 1e-15 (double)} relative — dependent columns whose remaining norms collapse
   mid-factorization by many orders. This is the guard's home turf: pivots must still
   land on independent columns first; rank detection must match the ε-known truth.
3. **Mass-cancellation**: all trailing columns nearly parallel to the first pivot
   (rank ~1 + tiny noise) — EVERY downdate cancels catastrophically at step 1; the
   guard must fire across the board. Assert Tier P + rank==1 detection at auto relTol.
4. **Gradual-decay attack** (defeats naive per-step-only guards): columns constructed
   so each single downdate is benign (ratio ~0.5) but cumulative decay over ~40 steps
   is ~1e-12 — exercises the vn2 cumulative test specifically. Construct via geometric
   0.95-style graded spectrum (randsvd) at n≥128.
5. **Ties**: duplicated-norm columns (exact duplicates and 1-ulp-apart norms). Tier P;
   also assert determinism (same input → same pivots, run twice).
6. **Scale extremes**: column norms spanning 1e±15 (double) / 1e±8 (float) within one
   matrix; all-zero columns; zero matrix; single column; n=1..3 tiny sizes.
7. **Fuzz sweep**: ≥64 random seeds, mixed shapes (square/tall, n 8..96), random rank
   deficiency injections; Tier P invariants on every one; Tier E on the seeds whose
   step-norm separation (measured by the reference) exceeds 8·sqrt(eps) at every step.
8. **Regression anchors**: the existing QRCP suite (rank-revealing tests, solveInPlace
   equivalence tests from commit 2/2.5, KnownValueRegression) must stay green untouched
   — they encode the current contract.

## Perf gate

Same-session A/B, QRVariantsBenchmark tall section: target QRCP.solveInPlace 2048×512
float ≤ ~1.5× QR.solveInPlace (from ~2.1×). If the win is smaller than ~15%, STOP and
report before merging — the added guard bookkeeping may be mis-shaped (e.g., scalar
per-column loop that should be vectorized).

## Open decisions (resolve during implementation, report in summary)

- **OQ-D1 (bit-compat)**: with downdating, R/Q bits may differ from the old path even
  on Tier-E inputs (norm VALUES feed only pivot CHOICE — if pivots match, the reflector
  math is untouched and factors should be bit-identical; verify and document which).
- **OQ-D2**: expose guard-fire count in RankInfo? RECOMMEND no (diag fields must be
  already-computed/meaningful; this is internal). Tests exercise the guard via
  constructed inputs, not counters.
- **OQ-D3**: `tol3z` literal per type via codegen (`math.sqrt(Consts.fProxyEpsilon)`
  hoisted method-local — remember template constants must be method-local).

## Pipeline

coder (kernel + cache + managed oracle + battery) → 2 adversarial reviews (one numerics:
guard correctness/overflow/swap bookkeeping; one testing: does the battery actually bite
— mutate the guard and confirm tests fail) → full suite → same-session perf A/B → commit
(not push).
