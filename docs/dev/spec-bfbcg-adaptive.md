# Mini-spec: adaptive `Krylov.bfbcg` — Cholesky-QR fast path for the orthonormalization step

## 0. Task

Make `Krylov.bfbcg` (`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.fProxy.cs`) adaptive:
on a well-conditioned search block, orthonormalize `P` via a cheap Cholesky-QR (Gram + Cholesky +
one triangular solve, all three already-existing vectorized kernels) instead of the full row-pivoted
`LQRP` factorization; fall back to `LQRP` (unchanged, still rank-revealing) only when Cholesky-QR's own
Gram is not comfortably well-conditioned. `bfbcg`'s public signature, recurrence, and robustness
guarantees are unchanged — only how `Phat = orth(P)` gets computed changes, per iteration, chosen by a
deterministic gate.

## 1. Context already read (do not re-derive from scratch)

- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.fProxy.cs` — the three shipped block
  solvers: ridge `bcg` (O'Leary, forms `PᵀAP` directly, Cholesky with ridge-retry via `BlockSolveSPD`, no
  orthonormalization), `bcgrq` (Dubrulle, orthonormalizes the RESIDUAL every iteration via `LQRP`), `bfbcg`
  (Ji & Li 2017, orthonormalizes the SEARCH block `P` every iteration via `LQRP`). This task touches
  **only `bfbcg`'s private helpers and core body** — `bcg` and `bcgrq` are not modified.
- `bfbcg`'s private helpers (all still needed, several reused unmodified):
  - `FactorLiveSearch(in fProxyMxN P, int sLive, int n, ref fProxyMxN Lbuf, ref fProxyMxN Pa) : int` —
    the existing `LQRP`-based orthonormalization: reads the live rows of `P` (read-only), writes the
    orthonormal `Phat` into `Pa`'s leading `sLive` rows (via `LQRP.decomp`, which itself can report a
    numerical rank `< sLive`), returns that rank. **Unmodified by this task** — it becomes the fallback
    branch of a new wrapper (§4.2).
  - `FactorGramOnce(in fProxyMxN G, ref fProxyMxN work, int r) : bool` — Cholesky-factors the `r x r`
    Gram `G = PhatᵀAPhat` once (ridge-retry safety net), so the same factor solves both `alpha` and
    `beta`. **Unrelated to this task** (operates on the already-orthonormal `Phat`'s `A`-Gram, `O(r³)`,
    not the `O(sLive²n)` bottleneck) — do not touch.
  - `BlockGram(in V, in W, ref G, int s)` — `G := V·Wᵀ`, symmetrized. Reused as-is for the NEW Gram
    `G = P·Pᵀ` (Cholesky-QR's Gram) by calling it with `V == W == the live P view`.
  - `View(in fProxyMxN buf, int m)` / `RowsView(in fProxyMxN buf, int rows)` — same-buffer narrower-shape
    reinterpretation (no allocation). Reused unmodified.
  - `CopyBlock(in fProxyMxN src, ref fProxyMxN dst, int s, int n)` — reused unmodified.
  - `LockConvergedRows`, `BlockScatterAddRows`, `BlockCTV`, `BlockAdd`, `BlockZplusT`, `BlockApplyPre`,
    `CountConverged` — untouched, not relevant to this task.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/CHO.fProxy.cs`:
  - `CHO.decompInPlace(ref fProxyMxN A_to_L) : DirectSolveInfo` — plain (unpivoted) Cholesky, in place.
    Returns `DirectSolveStatus.NotPositiveDefinite` on a non-positive pivot (checked *before* the
    `sqrt`, so failure never writes NaN/Inf); "the lower triangle is left partially overwritten -- treat
    A_to_L as destroyed on failure" (existing doc). Reused unmodified.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Blas.Triangular.fProxy.cs`:
  - `Blas.triLower(ref fProxyMxN L, ref fProxyMxN B_to_X) : DirectSolveInfo` (line ~175) — solves
    `L X = B` for `X`, **in place**, where `L.M_Rows == B_to_X.M_Rows` and each of `B_to_X.N_Cols` is an
    independent RHS. The implementation is already a row-recurrence with a unit-stride `UnsafeOP.axpy`
    over all `k = B_to_X.N_Cols` columns at once per row — this is **exactly** the shape our block
    convention needs: feeding `P` (`sLive` ROWS × `n` COLS, `L` = `sLive x sLive`) directly as `B_to_X`
    computes `Phat = L⁻¹P` row-by-row, each row-update vectorized across all `n` coordinates in one
    `axpy` call. **No new kernel is needed for the triangular solve** — this is the "custom kernel that
    computes only what is needed" the task asks for: it is an existing, already-optimized primitive,
    just newly applied to `P`'s own (not `A`'s) Gram. Reused unmodified.
  - PRECONDITION (documented on `triLower`): every `L[r,r]` must be nonzero, *unguarded*. This is why the
    gate (§4.1) must fully validate every pivot **before** calling `triLower`, never rely on it to detect
    a bad factor.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQRP.fProxy.cs` — `LQRP.decomp`'s unblocked Householder
  kernel (`lqrpKernel`): per step, a norm-based row pivot search, a guarded/downdated-norm bookkeeping
  loop (branchy, per-row `sqrt`/ratio/guard-trip re-sum), a Householder reflector generation + apply
  (rank-1-update structure, not a clean GEMM), **then a second, separate backward pass to reconstruct
  `Q`** from the stored reflectors. This is the ~2-3 `O(sLive²n)`-scale pass structure the benchmark's
  overhead comes from (§2) — none of it vectorizes as cleanly as the GEMM/TRSM-style kernels above.
  **Not modified by this task** — `bfbcg`'s fallback keeps calling `FactorLiveSearch` (hence `LQRP.decomp`)
  exactly as today.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/BlockSolveInfo.cs` — `//singularFile//` (copied once,
  not per-dtype). Return struct shared by `bcg`/`bcgrq`/`bfbcg`. New fields added here (§5) default to
  `0` for `bcg`/`bcgrq` automatically (unlisted fields in a `new BlockSolveInfo { ... }` object initializer
  are zero-valued) — **no edits needed to `bcg`'s or `bcgrq`'s return statements**.
- `Consts.fProxySqrtEps`, `Consts.fProxyEpsilon` — already used throughout this file and `LQRP.fProxy.cs`
  for tolerance/guard thresholds. No new `Consts` fields are needed (§4.1 derives its threshold from
  `Consts.fProxySqrtEps` via one extra `math.sqrt`, computed inline).
- `TestResults/benchmark-blockcg-sparse.txt` — BSR 2D Poisson (`arena.fProxyLaplacian2D`), independent
  random RHS (well-conditioned, NOT deliberately rank-deficient). Measured float-precision overhead of
  `bfbcg` over ridge `bcg`, same iteration count in every row (confirms the gap is pure per-iteration
  bookkeeping cost, not a convergence-rate difference):

  | N | s | bcg (ms) | bfbcg (ms) | overhead |
  |---|---|---|---|---|
  | 1024 | 2  | 3.3121   | 4.1399   | 25.0% |
  | 1024 | 4  | 5.1224   | 6.4103   | 25.1% |
  | 1024 | 8  | 8.4681   | 10.2776  | 21.4% |
  | 1024 | 16 | 12.6753  | 16.0069  | 26.3% |
  | 1024 | 32 | 20.8923  | 27.1156  | 29.8% |
  | 2304 | 2  | 17.4448  | 20.0075  | 14.7% |
  | 2304 | 32 | 107.9153 | 124.8433 | 15.7% |
  | 4096 | 2  | 56.4374  | 57.2406  | 1.4%  |
  | 4096 | 32 | 348.2135 | 390.7125 | 12.2% |

  (double-precision rows and the reported 12-37% headline range are in the same file; the float rows
  above are representative and sufficient for calibrating the estimates in §3.) Overhead grows with `s`
  at fixed `N` (bookkeeping-dominated regime) and shrinks as `N` grows at fixed `s` (matvec-dominated
  regime, since the BSR `ApplyBlock` cost is `O(nnz)` and dominates once large enough).
- `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BlockBFBCGTests.fProxy.cs` — existing test job
  (`BlockBFBCGTestJob`), 7 scenarios, structure mirrors `BlockCGrQTests.fProxy.cs`. Read in full before
  editing (§6 extends it, does not replace it).

## 2. Why this is the right amount of extra work (root-cause of the gap)

`bfbcg` orthonormalizes `P` via `LQRP.decomp` every iteration: a copy (`W.CopyFrom(in A)`), the unblocked
Householder reduction (row-pivot search + norm-downdate bookkeeping + reflector apply, `O(sLive²n)`, with
`sqrt`s and guard branches per row per step — not GEMM-shaped), and a **second**, separate backward sweep
to reconstruct `Q` (another `O(sLive²n)`-scale pass). Ridge `bcg` does none of this — it forms `PᵀAP`
directly off the raw (non-orthonormal) `P`. The fix in this spec (§4) replaces `LQRP.decomp` with
Cholesky-QR (`G = PPᵀ` via the already-GEMM-routed `BlockGram`, `O(r³)` Cholesky, one `Blas.triLower`
TRSM-style pass) **in the common well-conditioned case**, falling back to the existing `LQRP` path
untouched otherwise. This does not reach full parity with ridge `bcg` (Cholesky-QR is still one genuine
extra `O(sLive²n)` GEMM-class pass beyond what ridge `bcg` ever does — forming `P`'s own Gram is work
ridge `bcg` simply never performs, since it never orthonormalizes), but it replaces LQRP's 2-3
branch/sqrt-heavy passes with 2 GEMM/TRSM-class passes built from kernels this same file's own history
(`## Krylov.cg (BLOCK / multi-RHS)` DEVLOG entry) already measured as 5-9x faster than an equivalent
un-vectorized loop.

## 3. Option analysis — (A) skip-orth-entirely vs (B) cheaper orth kernel

### (A) Adaptive two-path: skip orthonormalization entirely on a well-conditioned Gram

Try forming `G = PᵀAP` directly on the **raw** (non-orthonormal) `P` and Cholesky-factoring it (exactly
ridge `bcg`'s own approach); if every pivot is comfortably above threshold, use O'Leary/ridge-style
coefficients directly (no `LQRP`, no Cholesky-QR, no orthonormalization pass at all — literally ridge
`bcg`'s per-iteration cost). Fall back to `LQRP`'s rank-revealing path only on a small/non-positive pivot.

**The blocking problem:** `bfbcg`'s recurrence is built around `Phat` (orthonormal) at every step — the
NEXT search block is defined as `P_{i+1} = Z_{i+1} + betaᵀ Phat_i` (built from the *orthonormal* basis,
not the raw candidate), and column-locking/deflation (`Live`, `sLive`, `LockConvergedRows`) is threaded
through that recurrence. Ridge `bcg`'s recurrence has no `Phat` concept at all, no per-column locking, and
a fixed block width `s` throughout. Switching the CARRIED STATE between these two representations
mid-solve — e.g. after several ridge-style (raw-`P`) iterations, deciding to switch into `bfbcg`'s
orthonormal-`Phat`-based recurrence, *while* some columns may already be locked out via `Live`/`sLive`,
which ridge `bcg`'s formulas have no concept of — requires either (a) restricting the fast path to only
ever run while `sLive == s` (nothing locked yet), which silently reduces its benefit on any solve where
early columns converge quickly (a common case, arguably the MOST common case for a well-conditioned
system), or (b) generalizing ridge `bcg`'s formulas to operate over an arbitrary live sub-block, which is
new, non-trivially-derived algebra, not a reuse of existing tested code. Either way this is real algorithm
design work with a real chance of a subtle correctness bug in the switch itself, not a kernel swap.

**Perf ceiling:** full parity with ridge `bcg` (recovers up to 100% of the gap when the fast path engages),
but only on iterations where it engages — and per the discussion above, "engages" would likely need to be
scoped down (e.g. "only before any column has locked") to keep the mode-switch tractable, which is exactly
the regime where iteration count is largest and matvec-bound, i.e. exactly where the gap is already
smallest (1.4% at N=4096, s=2). The scoped-down version of (A) would win least where the measured gap is
worst (small-N/large-s bookkeeping-dominated rows), because those are the rows most likely to have several
columns lock at different times during the solve.

**Verdict: reject for this task.** The mode-switch correctness risk is real and the practically-achievable
perf win (once scoped to something provably safe) is smaller than it first appears — it concentrates in
the regime that already has the smallest gap.

### (B) Cheaper orth kernel (Cholesky-QR) + gated LQRP fallback — chosen

Keep `bfbcg`'s recurrence **completely unchanged** — `Phat`, `G = PhatᵀAPhat`, `alpha`, `beta`, column
locking, all identical to today. Only how `Phat = orth(P)` is computed changes: try Cholesky-QR first
(§4.1), fall back to the existing `LQRP` path (§1) only when Cholesky-QR's Gram is not comfortably
well-conditioned. Both branches produce a valid orthonormal `Phat` (`Phat·Phatᵀ = I`) feeding the exact
same downstream formulas — **the recurrence cannot tell which branch ran**, so there is no mode-switch
risk at all, only a per-iteration choice of *how* to compute one intermediate value.

**Perf ceiling:** strictly less than (A)'s theoretical ceiling (Cholesky-QR is still one genuine extra
`O(sLive²n)` GEMM-class pass beyond ridge `bcg`'s per-iteration cost — see §2), but it applies uniformly,
every iteration, regardless of column-locking state, with zero new correctness surface. Estimated recovery
(kernel-constant-factor argument, §2, calibrated against this file's own "GEMM-routed 5-9x" precedent):
roughly 40-60% reduction of the *absolute* overhead. Applied to the measured rows: N=1024/s=32's 29.8%
gap should fall to roughly 12-18%; N=2304/s=32's 15.7% to roughly 6-9%; N=4096/s=32's 12.2% to roughly
5-7%; the already-small N=4096/s=2 row (1.4%) should land near zero. These are pre-implementation
estimates — confirm with the re-run in §7, do not treat the numbers above as a substitute for measuring.

**Verdict: recommended.** Zero mode-switch risk, reuses three already-tested, already-optimized primitives
(`BlockGram`, `CHO.decompInPlace`, `Blas.triLower`) with no new low-level kernel code, and the "custom
kernel that computes only what is needed" framing fits exactly: Cholesky-QR skips `LQRP`'s pivoting,
downdating bookkeeping, and Q-reconstruction pass — it computes only a Gram, a small Cholesky, and one
triangular solve.

### Do not build a hybrid

A hybrid (B's kernel as the default, with (A)'s "skip entirely while nothing has locked" bolted on for the
early iterations of every solve) was considered and rejected for this task: it reintroduces (A)'s
mode-switch surface (now bounded to "before the first lock," but still a real switch with its own
correctness argument to write and test) for an incremental win that, per the analysis above, mostly
overlaps with rows where the gap is already small. Not worth the risk in one coding session. Note it as a
possible future increment if benchmark data after this task shows the residual gap is still concentrated
in a regime a bounded (A) would help.

## 4. Chosen design (B)

### 4.1 `TryOrthonormalizeSearchFast` — Cholesky-QR attempt

```csharp
// Attempts Cholesky-QR orthonormalization of the live search block P (fast path): G = P Pᵀ (BlockGram),
// Cholesky-factor G in place, and -- only if every pivot clears a well-conditioned gate relative to the
// block's own norm scale -- forward-substitute Phat = L⁻¹P (Blas.triLower) into Pa's leading sLive rows.
// Returns false (Pa left untouched, P never modified) if the Gram is not positive-definite or any pivot
// is below gate, so the caller must fall back to the rank-revealing LQRP path. Never deflates on success
// (always reports full row rank sLive) -- rank-revealing behavior is LQRP's job alone.
static bool TryOrthonormalizeSearchFast(in fProxyMxN P, int sLive, int n, ref fProxyMxN Lbuf, ref fProxyMxN Pa)
{
    var Plive = RowsView(P, sLive);
    var G = View(Lbuf, sLive);                    // Lbuf reused as Gram scratch, then overwritten as L
    BlockGram(in Plive, in Plive, ref G, sLive);   // G := P Pᵀ

    fProxy diagMax = (fProxy)0;
    for (int i = 0; i < sLive; i++) { fProxy d = G[i, i]; if (d > diagMax) diagMax = d; }
    if (!(diagMax > (fProxy)0)) return false;      // all-zero live block -- defer to LQRP

    // Gate threshold: eps^(1/4) * sqrt(diagMax). Cholesky-QR's classical loss-of-orthogonality bound is
    // O(kappa(P)^2 * eps); it stays comfortably small (O(sqrt(eps))) only while kappa(P) stays well below
    // eps^(-1/2). Gating the smallest Cholesky pivot at eps^(1/4) * sqrt(diagMax) keeps kappa(P) below
    // roughly eps^(-1/4) -- deep inside the safe range, with margin, for a SINGLE Cholesky-QR pass (no
    // CholeskyQR2 needed). Treat eps^(1/4) as the starting point, not a hard requirement: if the
    // benchmark/tests in this spec show the fast path is engaging too rarely to close the measured gap
    // (§3), a looser exponent (e.g. eps^(1/3), or plain Consts.fProxySqrtEps) is safe to try -- keep the
    // "constant * sqrt(diagMax)" shape (a fixed, deterministic function of the Gram) either way.
    fProxy gateTol = math.sqrt(Consts.fProxySqrtEps) * math.sqrt(diagMax);

    var info = CHO.decompInPlace(ref G);           // G: Gram -> L (destructive, single pass, no ridge)
    if (info.status != DirectSolveStatus.Success) return false;
    for (int i = 0; i < sLive; i++)
        if (!(G[i, i] >= gateTol)) return false;   // near-singular relative to the block's own scale

    var PaLive = RowsView(Pa, sLive);
    CopyBlock(in Plive, ref PaLive, sLive, n);      // P is never mutated by this path
    Blas.triLower(ref G, ref PaLive);               // PaLive: P -> L^-1 P == Phat

    return true;
}
```

Notes:
- `Lbuf` is the SAME `s x s` scratch buffer `FactorLiveSearch`/`LQRP.decomp` already use for `L` — reused
  here as Cholesky-QR's Gram-then-factor scratch. Whichever branch (fast or fallback) actually runs fully
  overwrites its own `sLive x sLive` prefix, so there is no stale-data hazard between the two uses.
- `P` is read-only throughout (`in fProxyMxN P`) — the raw search block is copied into `Pa` *before*
  `Blas.triLower` mutates it in place, so `P`'s content survives untouched regardless of which branch
  ran, matching `FactorLiveSearch`'s existing contract exactly (drop-in replacement, §4.2).
- No new low-level kernel: `BlockGram`, `CHO.decompInPlace`, `Blas.triLower`, `CopyBlock` are all reused
  unmodified.

### 4.2 `FactorLiveSearchAdaptive` — the gated wrapper (replaces both `FactorLiveSearch` call sites)

```csharp
// Orthonormalizes the live search block, trying the cheap Cholesky-QR fast path first
// (TryOrthonormalizeSearchFast) and falling back to the rank-revealing FactorLiveSearch (LQRP) only when
// the fast path declines. Same contract as FactorLiveSearch: writes the orthonormal result into Pa's
// leading rows, returns its rank. `tookFastPath` reports which branch ran (BlockSolveInfo path counters).
static int FactorLiveSearchAdaptive(in fProxyMxN P, int sLive, int n, ref fProxyMxN Lbuf, ref fProxyMxN Pa,
                                     out bool tookFastPath)
{
    if (TryOrthonormalizeSearchFast(in P, sLive, n, ref Lbuf, ref Pa))
    {
        tookFastPath = true;
        return sLive;
    }
    tookFastPath = false;
    return FactorLiveSearch(in P, sLive, n, ref Lbuf, ref Pa);
}
```

### 4.3 Integration into `bfbcg<TOp, TPre>`

Exactly two call sites change, both currently reading `int r = FactorLiveSearch(in P, sLive, n, ref Lbuf,
ref Pa);` — one in the setup block (`r0`), one at the tail of the loop body (`saNew`). Replace each with
`FactorLiveSearchAdaptive`, capture `tookFastPath`, and increment one of two local counters declared
alongside the existing `minActive`/`saSearch` locals near the top of the method body:

```csharp
int fastPathOrthCount = 0, slowPathOrthCount = 0;
...
// setup:
int r0 = FactorLiveSearchAdaptive(in P, sLive, n, ref Lbuf, ref Pa, out bool fast0);
if (fast0) fastPathOrthCount++; else slowPathOrthCount++;
...
// loop tail:
int saNew = FactorLiveSearchAdaptive(in P, sLive, n, ref Lbuf, ref Pa, out bool fastK);
if (fastK) fastPathOrthCount++; else slowPathOrthCount++;
```

Every other line of `bfbcg` (argument validation, the alpha/beta solves via `FactorGramOnce`, column
locking, the X/R updates, the cleanup block) is **unchanged**. Both counters flow into the final
`BlockSolveInfo` at every exit path (the shared `cleanup:` label already funnels every `goto cleanup`
through one `return new BlockSolveInfo { ... }` — add `fastPathOrthCount = fastPathOrthCount,
slowPathOrthCount = slowPathOrthCount` there).

This **replaces `bfbcg`'s body in place** (per the task's deliverable #2) — `bfbcg` itself becomes
adaptive, its public overload ladder (all 8 overloads) is untouched, and there is now exactly one
fast-and-robust block-CG variant alongside `bcgrq` (still LQRP-always, untouched) and ridge `bcg`
(untouched, candidate for later retirement — §8).

## 5. `BlockSolveInfo` change

Add two fields to `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/BlockSolveInfo.cs` (a `//singularFile//`
— one edit, not per-dtype):

```csharp
/// <summary>Number of per-iteration search-block orthonormalizations that used the cheap Cholesky-QR
/// fast path. Meaningful only for <see cref="Krylov.bfbcg"/> -- always 0 for <see cref="Krylov.bcg"/>
/// (never orthonormalizes) and <see cref="Krylov.bcgrq"/> (always uses the rank-revealing path).</summary>
public int fastPathOrthCount;

/// <summary>Number of per-iteration search-block orthonormalizations that fell back to the
/// rank-revealing LQRP path (near-singular/ill-conditioned Cholesky-QR Gram, or a genuinely
/// rank-deficient block). Meaningful only for <see cref="Krylov.bfbcg"/> -- always 0 for
/// <see cref="Krylov.bcg"/>; for <see cref="Krylov.bcgrq"/> this concept does not apply (it has no fast
/// path to fall back FROM) and this field stays 0 there too.</summary>
public int slowPathOrthCount;
```

`fastPathOrthCount + slowPathOrthCount` always equals the total number of orthonormalization calls made
during a `bfbcg` solve (self-consistent by construction — every `FactorLiveSearchAdaptive` call increments
exactly one of the two). Optionally include both in `ToFixedString()`'s summary; not required.

## 6. Tests — extend `BlockBFBCGTests.fProxy.cs`

All 7 existing scenarios (`MatchesScalarCgPerColumn`, `KnownSolutionRecovered`, `BlockAdvantageIterations`,
`RankDeficientDeflates`, `PreconditionedMatchesScalar`, `IdentityFoldMatchesUnpreconditioned`,
`NeverWorseThanRidge`) must keep passing unmodified except where noted below — they are exactly the
"adaptivity doesn't change results to tolerance" check the task asks for: none of them special-case an
implementation path, so their continuing to pass at the existing tolerances (`Tol()`, `ResidualSlack()`)
against the new adaptive body **is** that acceptance signal.

New/changed cases:

1. **Strengthen `RankDeficientDeflates`** (existing test, add one assertion): after the existing
   `Assert.IsTrue(info.minActive < info.rhs)`, add `Assert.IsTrue(info.slowPathOrthCount > 0)` — the
   rank-1-in-`B` scenario must genuinely engage the LQRP fallback, not merely happen to report a small
   `minActive` for an unrelated reason.
2. **New `WellConditionedStaysOnFastPath`** — reuse `MatchesScalarCgPerColumn`'s system-building style
   (`BuildDenseSPD`, well-conditioned, `n=20, s=4` or similar). Run `bfbcg`, assert `info.Solved`, assert
   `info.slowPathOrthCount == 0`, assert `info.fastPathOrthCount > 0` (and, as a sanity bound tying it to
   the iteration count without over-specifying the exact early-exit arithmetic, `info.fastPathOrthCount >=
   info.iterations`).
3. **New `IllConditionedNotRankDeficientTriggersFallback`** — the specific scenario that justifies the
   counter fields over relying on `minActive` alone (§1: an ill-conditioned-but-full-rank Gram can still
   trip the Cholesky-QR gate while `LQRP` itself finds full rank, so `minActive` would stay `== rhs`
   despite the fallback having run). Use `BuildStretchedSPD` (already in the test file) with a `condSpan`
   large enough to trip the gate derived in §4.1 but not so large that the solve fails to converge within
   a generous `maxIter` — tune this empirically (start near the `condSpan = 8` already used by
   `NeverWorseThanRidge`; increase if the gate does not trip). Assert `info.Solved`,
   `info.slowPathOrthCount > 0`. Do **not** assert anything about `minActive` here (it may legitimately
   equal `rhs`) — that is the point of this test.
4. **New `FastPathOrthonormalityHolds`** — on the `WellConditionedStaysOnFastPath`-style system, after a
   solve confirms `slowPathOrthCount == 0`, independently re-derive one fast-path `Phat` (call
   `Krylov`'s internal helpers is not available cross-file to a test — instead, verify indirectly: the
   existing `MatchesScalarCgPerColumn`/`KnownSolutionRecovered`-style forward-error/residual checks
   already certify correctness end-to-end; add a residual-only check here at a tighter tolerance than the
   existing `Tol()` if convenient, but do not require reaching into `Krylov`'s private helpers from the
   test). If reaching the private helper is impossible without new public surface, treat this bullet as
   satisfied by #2 plus the existing accuracy assertions in `MatchesScalarCgPerColumn`; do not add new
   public API only to unit-test an implementation detail.
5. Re-run `NeverWorseThanRidge` unmodified — its `ResidualSlack()`-scaled comparison against ridge `bcg`
   should keep passing (both solvers are unaffected by this change in the ill-conditioned regime it
   already exercises: `bfbcg` will mostly take the LQRP fallback there anyway).

Use `Assert.IsTrue(bool)` / `Assert.AreEqual` only, never the string-message overload (BC1071 -> silent
Mono fallback, per this file's own `## Krylov.cg` DEVLOG entry).

## 7. Benchmark acceptance — re-run `BlockCGSparseBenchmark`, no code changes required

`bfbcg`'s public signature is unchanged, so
`Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/BlockCGSparseBenchmark.fProxy.cs` and
`Assets/LinearAlgebra/Benchmarks/BlockCGSparseBenchmark.cs` need **no edits** — the existing `bfbcg` row
already exists in the sweep. Re-run it (`Tools/benchmark.ps1` or the project's existing benchmark
invocation) and compare the new `bfbcg` row against `bcg` at every `(grid, s)` in
`TestResults/benchmark-blockcg-sparse.txt`. Acceptance (two-tier, per §3's honest estimate — do not
require a single uniform bound that the small-N/large-s corner cannot plausibly hit in one pass):

- **Matvec-dominated rows** (`N >= 2304`): `bfbcg` overhead over `bcg` should land within roughly 5-8% (down
  from the current 10-16% at N=2304 and 1-12% at N=4096).
- **Bookkeeping-dominated rows** (`N = 1024`, `s >= 8`): overhead should be **measurably reduced from the
  current baseline** (roughly half or better — e.g. N=1024/s=32's current 29.8% should fall to
  approximately 12-18%), but is not required to reach single digits — see §3 for why full parity in this
  regime is out of scope for approach (B).
- `bcgrq`'s row must be numerically unchanged (within normal run-to-run noise) — it is not touched by this
  task.
- Iteration counts (the `iters` column) for `bfbcg` must stay identical to the pre-change baseline at every
  row (this change only alters *how* `Phat` is computed each iteration, not the recurrence itself, so the
  convergence trajectory should not shift outside ordinary floating-point-order noise).

If the matvec-dominated tier does not land in range, the gate threshold (§4.1) is the first thing to
revisit (too conservative -> fast path rarely engages); if the well-conditioned tests (§6) start failing
instead, the threshold is too loose.

## 8. Note on ridge `bcg` retirement safety (task #24)

Once `bfbcg` is adaptive and (a) the benchmark acceptance in §7 shows it close to `bcg`'s speed on
well-conditioned sweeps, and (b) all existing rank-deficiency/robustness tests (§6, unchanged) keep
passing, retiring ridge `bcg` becomes safe from a **capability** standpoint — `bfbcg` would be both fast
enough and strictly more robust (never ridge-patches, always either fully orthonormalizes cheaply or
falls back to rank-revealing deflation). It is **not** automatically safe from an **API-surface**
standpoint: `bcg`'s 8 public overloads are called directly by `BlockCGTests.fProxy.cs`,
`BlockCGBenchmark.fProxy.cs`, and `BlockCGSparseBenchmark.fProxy.cs` today, and those call sites would need
to be migrated or the tests/benchmarks retired alongside `bcg` itself. That migration is task #24's own
scope, not this task's — this spec only removes the *performance* reason to keep `bcg` around; it does not
touch `bcg` or any of its callers.

## 9. Implementation checklist (ordered)

1. Add `TryOrthonormalizeSearchFast` and `FactorLiveSearchAdaptive` as new private helpers in the `bfbcg`
   section of `Krylov.Block.fProxy.cs` (§4.1, §4.2), placed near the existing `FactorLiveSearch`/
   `FactorGramOnce` helpers.
2. Add `fastPathOrthCount`/`slowPathOrthCount` fields to `BlockSolveInfo.cs` (§5).
3. Swap `bfbcg<TOp, TPre>`'s two `FactorLiveSearch` call sites for `FactorLiveSearchAdaptive`, thread the
   two counters through to every `return new BlockSolveInfo { ... }` (§4.3). Do not touch anything else in
   the method body.
4. Regenerate (`Tools/regen.ps1`) and confirm float + double both compile clean.
5. Extend `BlockBFBCGTests.fProxy.cs` per §6 (one strengthened assertion, two-to-three new test cases).
6. Run the full suite headlessly; confirm the exact line `Result=Passed total=N passed=N failed=0` (never
   pipe through `| tail`).
7. Re-run `BlockCGSparseBenchmark` (§7); compare against `TestResults/benchmark-blockcg-sparse.txt`'s
   existing numbers. If the gate threshold needs tuning to hit §7's targets, iterate on the constant in
   `TryOrthonormalizeSearchFast` only (keep the `constant * sqrt(diagMax)` shape) and re-run tests +
   benchmark again.
8. Add a `## Krylov.bfbcg` DEVLOG.md entry (newest-first, dated) in
   `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/DEVLOG.md` capturing: the chosen gate constant and why,
   the measured before/after benchmark numbers, and the (A)-vs-(B) decision summary from §3 (do not
   duplicate this spec's full reasoning — one or two dense paragraphs, per the project's DEVLOG format).
   Do not put any of this in code comments (contracts only, per `CLAUDE.md`).

## 10. Determinism

The gate in §4.1 (`G[i,i] >= gateTol`) is a data-dependent branch, exactly like three branches this same
file already ships with today: `BlockSolveSPD`'s ridge-retry ladder (branches on whether `CHO.decompInPlace`
succeeds), `LQRPRank`'s rank cutoff (`|L[i,i]| > tol`), and `LockConvergedRows`' convergence test
(`rr <= thr[orig]`). Under the project's `FloatMode.Strict` discipline (fixed reduction order for
`+ - * / sqrt`, no reassociation — see the "Determinism analysis" project note), `BlockGram`, the Cholesky
pivots, and the gate comparison are all built from the same fixed sequence of IEEE-754 operations on the
same input bits, so for a given `P` the branch outcome is **bit-identical across architectures** — this is
not a new determinism risk, it is the same posture already accepted for the three existing gates above.
Do not add extra synchronization/guarding beyond what those existing helpers already do; do not attempt to
make the gate "softer" (e.g. blending both paths) in the name of determinism — that would be solving a
problem this codebase has already decided is a non-problem under Strict mode. The one genuine (pre-existing,
not newly introduced) caveat: outside `FloatMode.Strict` or across SIMD widths where Burst permits
reassociation, ULP-level differences near the threshold boundary could in principle flip the branch on one
arch and not another; if that ever needs closing off project-wide it is a determinism-harness concern (see
the project's determinism conformance harness work), not something this task should attempt to solve
locally.

## 11. Acceptance criteria

- `TryOrthonormalizeSearchFast` and `FactorLiveSearchAdaptive` exist in `Krylov.Block.fProxy.cs`, generated
  cleanly for both `float` and `double`, with the exact contracts of §4.
- `FactorLiveSearch`, `FactorGramOnce`, `BlockGram`, `CHO.decompInPlace`, `Blas.triLower`, `CopyBlock` are
  reused **unmodified** (no signature or body changes to any of them).
- `bfbcg`'s public overload ladder (all 8 overloads) is byte-identical to before this task.
- `bcg` and `bcgrq` (their bodies and all overloads) are byte-identical to before this task.
- `BlockSolveInfo` gains exactly the two fields in §5; every existing field/method on it is unchanged.
- All 7 pre-existing `BlockBFBCGTests` scenarios pass, including the strengthened `RankDeficientDeflates`
  (§6.1).
- The new `WellConditionedStaysOnFastPath` and `IllConditionedNotRankDeficientTriggersFallback` tests
  (§6.2, §6.3) exist and pass.
- Full project test suite green: the literal line `Result=Passed total=N passed=N failed=0` from the
  headless test run, `N` including the new/strengthened bfbcg tests, `failed=0`.
- `BlockCGSparseBenchmark`'s re-run numbers meet §7's two-tier targets (or the gate threshold has been
  tuned per §9 step 7 until they do); `bcgrq`'s row is unaffected.
- No edits to `README.md`. No edits to anything under `Assets/LinearAlgebra/Source/` or
  `Assets/LinearAlgebra/Benchmarks/Generated/` (generated output — regenerate instead).
- No edits to `Pivot/Pivot.cs`, `Pivot/Pivot.Operations.cs`, `LU.fProxy.cs`, `Solvers.fProxy.cs`, or any
  `fProxyMxN`/`fProxyN` `CopyTo`/`CopyFrom` members (unrelated priority-backlog items, out of scope here).

## 12. Out of scope (do not do these in this task)

- Approach (A) (skip-orthonormalization-entirely mode switch) or any hybrid reintroducing it — rejected in
  §3; revisit only as a separate future task if §7's benchmark re-run shows the bookkeeping-dominated
  residual gap is still unacceptably large after this task ships.
- `CholeskyQR2` (a second re-orthonormalization pass for extended stability range) — the single-pass gate
  in §4.1 is chosen conservatively enough that it should not be needed; note as a follow-up only if
  `FastPathOrthonormalityHolds`-style testing (§6.4) reveals real orthogonality drift.
- Any change to `bcg` (ridge block-CG) or its callers/tests/benchmarks, including the task #24 retirement
  itself (§8) — this task only makes that retirement *safe*, it does not perform it.
- Any change to `bcgrq` — untouched, per the task statement.
- Any change to `LQRP.fProxy.cs`, `CHO.fProxy.cs`, or `Blas.Triangular.fProxy.cs` kernels themselves — all
  reused exactly as they exist today.
- Resolving the `Pivot` "Arena dependency?" TODO, `LU`/`Solvers` consolidation, `IMatrix.CopyTo`/`CopyFrom`
  stubs, SVD, least squares, optimizers, sparse-matrix work, or View/Slice — unrelated priority-backlog
  items, not touched here.
- Adding a dense (non-BSR) benchmark variant, or any new benchmark file — the existing
  `BlockCGSparseBenchmark` sweep is reused as-is (§7).
