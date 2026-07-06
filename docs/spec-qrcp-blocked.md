# Spec: Blocked QRCP (dlaqps-style level-3 panels)

STATUS: APPROVED 2026-07-06 (user: "yes ofc, blocked QRCP interests me").
QUEUED behind the SVD/Eigen convergence overhaul (Unity-lock serialization).
Prerequisite landed: norm downdating + vn1/vn2 cumulative guard (`e865f27`).

## Goal

Raise QRCP factorization from level-2 (per-reflector full trailing update) to LAPACK
dgeqp3/dlaqps-style partially-blocked level-3. Current: `QRCP.solveInPlace` 2048×512
float 63.9 ms ≈ 2.0× QR's 32 ms. Target: **≤ 1.5× QR** (~48 ms), realistic landing zone
1.3–1.6×. This is where the remaining gap actually lives — `e865f27`'s A/B proved norm
maintenance was only ~15% of runtime; the dominant cost is the unblocked trailing updates.

## Why blocking is possible now

Pivot selection needs current column norms, NOT current column data. Downdated vn1 gives
the norms without touching the trailing matrix, so trailing updates can be deferred and
batched into one GEMM per panel. Downdating (already landed) is the enabler; this spec is
the payoff.

## Algorithm (dlaqps mapped to our conventions)

Outer loop = panels of width NB. Per panel, maintain `F` sized (n − panelStart) × NB —
the FULL remaining width, panel columns included: F's panel-column rows are zero in the
direct term but filled by the correction term, and step 2 consumes them. Invariant: after
k reflectors, the not-yet-updated part satisfies A_true = A_stale − V·Fᵀ.
(Verified against the reference dlaqps.f source 2026-07-06; ranges below use 0-based
inclusive `..` notation to avoid off-by-ones.)

Per step k within a panel (k = 0..kb−1, panel-local):
1. Pivot by max vn1 over remaining columns; swap columns j↔piv in: A (full column,
   strided), **R's already-written prefix — ALL rows factored so far, previous panels
   included, not just this panel's** (our split Q/R storage — LAPACK swaps one combined
   column; we swap both pieces), F **rows** (the k already-filled columns), vn1/vn2, P.
   Tie tolerance unchanged from e865f27.
2. Bring ONLY the pivot column up to date w.r.t. the previous reflectors:
   `A(:,k) −= V(:, 0..k−1)·F(k, 0..k−1)ᵀ` — O(m·k). Reads F row k (a panel-column row —
   why F must cover the panel columns).
3. Generate Householder u_k (τ≡1 convention, same as blocked QR core).
4. Compute F column k WITHOUT updating the trailing matrix:
   direct term `F(j,k) = A_staleᵀ·u_k` for trailing j only (zero for panel-prefix rows),
   then correction over ALL rows: `F(:,k) −= F(:, 0..k−1)·(V(:, 0..k−1)ᵀ·u_k)` — one GEMV
   pass over the stale trailing matrix plus a k-sized correction (the Vᵀu vector is the
   AUXV scratch, length NB). Row-major-friendly: Aᵀu is the vecMatDot axpy pattern.
5. Update row k of the trailing part incrementally (needed for R and for the vn1
   downdate): `A(k,j) −= V(k, 0..k)·F(j, 0..k)ᵀ` for trailing j — O(n_t·k). **Range is
   0..k INCLUSIVE of the reflector just generated** (its contribution to row k comes via
   F(:,k)) — LAPACK uses all K columns here vs K−1 in step 2; dropping the newest one is
   the natural off-by-one.
6. Downdate vn1 with the e865f27 guard formula. **On guard trip: do NOT re-sum — mark.**
   The trailing matrix is stale mid-panel, so an immediate exact re-sum (what the
   unblocked e865f27 kernel does — do NOT copy that pattern here) would compute a WRONG
   norm. LAPACK leaves vn1(j) un-downdated (stale) and appends j to a marked list; we use
   an explicit mark buffer, NOT LAPACK's hack of threading a linked list through vn2
   (swap-unsafe and obscure). Finish the current step, then cut the panel short
   (kb = k+1 < NB) — dlaqps returns kb to dgeqp3 for exactly this.
7. Panel end (full or cut): one GEMM `A(rows rk+1.., cols kb..) −= V·Fᵀ` (rank-kb; row rk
   is already current from step 5), THEN exact re-sum of the marked columns over the
   now-updated trailing matrix, setting vn2 = vn1 for them.

Memory-traffic story (why this wins): unblocked spends 2 passes over the trailing matrix
per column (GEMV for wᵀ=uᵀA, then rank-1 write-back). Blocked spends 1 read-only GEMV
pass per column (step 4) + 1 GEMM pass per kb columns — same FLOPs, but half of them move
into the ~70 GFLOP/s GEMM ceiling instead of the memory-bound rank-1 path.

## Structural

- `fProxyQRCPCache` gains: F buffer (n×NB upper bound, row-major — row-major F makes both
  the step-4 correction GEMV (contiguous F rows) and the flush kernel (per-A-row dot
  against contiguous F rows, kb ≤ NB fits registers) cache-friendly), the AUXV scratch
  (length NB), and the guard mark buffer; vn1/vn2 stay. Raw overloads keep Temp-allocating
  a cache internally.
- Storage landmine: LAPACK keeps reflector V in A's lower triangle; our `A_to_Q`
  progressively becomes Q instead. The panel's V must be staged the same way
  `qrDecompositionBlockedCore` already stages it — mirror that, don't invent a scheme.
- NB: method-local const (codegen: no class-level consts), start at 32 = QR_BLOCK.
- Size gate mirroring QR: blocked core only when n ≥ 2·NB; below that, keep the current
  (e865f27) unblocked path verbatim. Both `decomp*` (forms Q) and `solveInPlace` (no Q
  reconstruction) must route through the blocked core — mirror how QR's blocked core
  serves both.
- Reuse the wy*/pointer-GEMM helpers in UnsafeOP (wyVtC/wySubVW pattern: V lives as
  strided columns inside A, Vp+Vld pointers) where they fit; F·(Vᵀu) and V·Fᵀ are new
  small kernels if the existing ones don't map.

## Testing (the battery is the acceptance gate, again)

- **Entire existing QRCPDowndateTests battery must pass through the blocked path** — the
  size gate means battery sizes ≥ 2·NB exercise it; verify at least Kahan n=64, fuzz,
  mass-cancellation, gradual-decay, scale extremes hit the blocked core (add sizes if a
  case tops out below the gate).
- Blocked-vs-unblocked equivalence: Tier E (well-separated) — identical PIVOT SEQUENCE +
  Q/R agreement to a tight tolerance (see OQ-B2: bit-identity is NOT expected); Tier P —
  invariants only (|R| diag non-increasing, A·P == Q·R, rank).
- Guard-trip-mid-panel: mass-cancellation at n ≥ 2·NB must cut panels and still satisfy
  all invariants; assert via the existing oracle.
- Panel-boundary edges: n = 2·NB, 2·NB+1, kb cut to 1, rank collapse inside a panel,
  m = n vs tall.
- Existing QRCPTests regression anchor untouched and green.

## Perf gate

Same-session stash A/B on QRVariantsBenchmark, quiet machine (QR control must reproduce
~31-32 ms float tall and stay stable run-to-run). Gate: ≥ 25% improvement over e865f27's
63.9 ms at 2048×512 float (target zone ≤ 52 ms), else stop-and-report before committing.
Also record 2048×1024 and square 512/1024 for the docs.

## Open questions

- OQ-B1: NB = 32 assumed (QR precedent). If gate is missed, sweep 16/48/64 before
  concluding — pivoted panels have different cache behavior than plain QR panels.
- OQ-B2: expectation set deliberately LOW — bit-identity vs unblocked will almost
  certainly NOT hold (trailing values arrive via GEMM accumulation vs a chain of rank-1
  updates = different summation order; same reason the blocked QR path isn't bit-identical
  to its unblocked small-size path, an already-accepted precedent in this codebase). The
  Tier-E contract is: identical pivot sequence (pivoting keys off well-separated vn1, so
  last-bit vn1 differences can't flip it) + Q/R within a tight tolerance. If bit-identity
  happens to hold empirically, tighten the assert and celebrate; don't chase it.
  Note the guard never fires on Tier-E inputs, so the cut-short path can't affect this.
- OQ-B3: guard-trip flush = simple two-pass (GEMM flush, then batched re-sum) first;
  fusing is a follow-up only if profiling says it matters.
- OQ-B4: does `decompInPlace`'s R-column swap interact with callers that alias R into
  A-adjacent storage? Audit the three decompInPlace overloads' contracts before touching.
