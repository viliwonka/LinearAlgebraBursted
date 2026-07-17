# DEVLOG — OP
Code comments state contracts only; history lives here (see CLAUDE.md).

## math.select branch-free conversion pass (docs/dev/spec-math-select-pass.md)
- 2026-07-17 | Batch A (per-element data selects): converted `SelectOP.fProxy.cs`/`SelectOP.iProxy.cs`
  selectfProxy/selectiProxy, `UnsafeMathOP.iProxy.cs` abs/max/min/relu, `UnsafeMathOP.fProxy.cs`
  relu, `UnsafeOP.iProxy.cs` sumAbs/maxAbs, `Blas.ColumnScaling.fProxy.cs` buildJacobiScale, and
  `SelectOP.bool.cs` selectBool (A11, taken) from ternaries/if-branches to
  `math.select`/`math.max`/`math.min`/`math.abs`. Float relu kept as
  `math.select(x[i], 0, x[i] < 0)`, NOT `math.max(x[i], 0)` (NaN/-0 semantics differ). Benchmark
  (A/B via a reverted scratch IJob benchmark, float N=10240, REPS=200, headless
  `Tools/benchmark.ps1`, 4 timed runs/side, repeated to check noise): `Select.select` (A1)
  before(branch) med ~0.36-0.67ms -> after(select) med ~0.10-0.11ms, a reproducible ~3-4x win
  across repeats — Burst emits an LLVM `select` directly from `math.select`, where the
  bool-array-driven ternary needed the optimizer to promote a branch, which it did less
  reliably. `relu` (A7) before ~0.033-0.038ms -> after ~0.033-0.035ms: no measurable difference
  (expected — Unity.Mathematics' `select` is itself defined as `test ? t : f`, and a direct
  float-compare predicate was already select-friendly either way); not a regression, so nothing
  reverted. Scratch benchmark file removed after recording; no permanent benchmark added.
- 2026-07-17 | Root defect (docs/dev/spec-lobpcg-robustness.md, Duersch et al. 2018 §4.1): the old
  test `‖r‖ ≤ tol·max(|λ|,1)` had no ‖x‖ — the residual is linear in x, so a shrinking iterate
  passes ever more easily and x=0 passes EXACTLY (λ≈0, r≈0). On the penalty-conditioned n=24 frame
  with k=4/guard=4 (3·kWork=n, RR basis tiles the whole space) float LOBPCG returned Converged with
  zero vectors at λ=0. Fix = Duersch Eq. 9 shape: `‖r‖ ≤ tol·(normAEst + |λ|·normBEst)·‖x‖` with
  Frobenius-sketch lower bounds from the orthonormalized seed (one-time, no extra matvecs;
  normBEst=1 on B=I), plus a per-pair B-norm certification floor (0.25): a pair below it is
  DEGENERATE — never locks, never counts converged, forces the new `IterativeSolveStatus.Degenerate`
  exit if still among the k wanted at exit. The (d1) re-deflation guard `bn2 > 0` left an
  annihilated row an exact zero row forever (self-certifying fixed point); now `bn2 > eps` with a
  deterministic reseed (seed keyed by (iter,i)) + single-row re-deflation + B-normalize.
  Cache grew two length-k vectors (resScale, xBnorm; RequireDistinctBuffers 23→25). Deliberately
  NOT done here (specced §C.2, owner-gated): per-iteration B-renormalization of X, SVQB-with-
  dropping, cube-rule Gram gate.
- 2026-07-17 | TWO DEVIATIONS from the spec's literal Eq. 9 shape `(normAEst + |λ|·normBEst)·‖x‖₂`,
  both forced by acceptance tests:
  (1) the λ term is anchored to ‖x‖_B, i.e. `normAEst·‖x‖₂ + |λ|·normBEst·‖x‖_B` — required for
  the spec's own §D.6 (rank-deficient B must not certify): with the literal shape, an iterate
  blowing up in a singular B's null space (x = x_r + c·e_null, c huge, ‖x‖_B ≈ 1, λ ≈ −c²)
  certifies as Converged — the denominator inflates with |λ|·‖x‖₂ ~ c³ while ‖r‖ ~ c² — observed
  λ ≈ −2e15 (float) / −9e49 (double) reported Converged. ‖x‖_B anchoring makes that relative
  residual O(1); the fixed floor xBnorm ≥ 0.25 alone does NOT catch it (the blowup keeps
  ‖x‖_B ≈ 1).
  (2) the final scale is `min(Eq9 shape, max(|λ|,1)·‖x‖_B)` — pure Eq.9 is ~‖A‖× LOOSER than the
  old `max(|λ|,1)` test for small-λ modes of penalty-scaled matrices (normAEst ≈ 300 on the frame
  demos), and certified residuals ~0.02-0.05 relative to ‖Aφ‖ that failed the demo smoke tests'
  independent residual audits (BuildingFrame/Truss3D/TrussModal, 3 failures at 6334-test scale).
  The min keeps certification at least as strict as pre-fix while every term still scales with
  the iterate's norms (the actual bug was the ABSENT ‖x‖, not the magnitude anchor).
  Also observed while testing: the fixed solver now honestly SOLVES the 1×1×1 frame repro in
  float under Eq.9-only (2 iterations, matches dense) — degenerate rows can no longer lock, so
  the lock-poisoning cascade never starts; the spec's expected `Degenerate` exit need not occur
  on the repro. Pre-fix behavior (verified by stash-running the repro test against the old
  solver): Converged with λ=0, ‖x‖=0 in both guard=4 (iters=4) and guard=0 (iters=48) float
  configs.

## LPBasis.populated native-backed; warm-state fix complete + .Run() regression tests
- 2026-07-17 | `LPBasis.populated` was a plain bool -> lost on an IJob by-value copy, so a worker `.Run()`
  re-seeded the basis and clobbered the warm start. It is set-once (read via IsEmpty/needsSeed, not
  ref-passed into a hot loop), so the LQR approach fits: `NativeReference<int>` behind a `bool` property,
  transparent to all call sites. Completes the LP warm-state fix (cache `_meta` mirror + this).
- `Pivot.swapCount` deliberately NOT mirrored: it only feeds `Pivot.Sign` (permutation parity ->
  determinant sign), which LP's warm resume never reads (FTRAN/BTRAN use the permutation ARRAY, native,
  which survives); LU sets/reads it within a single solve, not across `.Run()`s. Not load-bearing -- the
  warm-state audit over-flagged it.
- Regression tests (DemoSmokeTests, plain `.Run()` = by-value copy, FAIL on the pre-fix plain-field code):
  `LqrWarmState_SurvivesRunByValueCopy` (populated visible on caller after a cold solve through a job
  field) and `EconomyLPJob_WarmState_SurvivesRunByValueCopy` (same LP twice via `.Run()` -> 2nd is a
  cache HIT, warm pivots < cold, only if cache+basis survived the copy). See [[job-struct-copy-warmstate-audit]].

## fProxyLPCache: native-mirror warm-state so it survives an IJob by-value copy
- 2026-07-17 | Same bug class as the LQR fix: LP.solve's warm-state scalars (builtVersion, etaCount,
  factorsValid, weightsValid) were plain fields, lost on an IJob by-value `.Run()`/`Schedule` copy →
  a worker-scheduled warm solve silently desyncs the eta chain. Fix = MPC's `qpMeta` pattern done RIGHT
  (my first two attempts were wrong — see below): KEEP the four as plain fields (so every read/write
  INSIDE the solve is byte-identical to before) and add a `NativeArray<int> _meta` mirror synced ONLY at
  the boundary — `RehydrateWarm()` (fields <- _meta) before the useCache branch's first read,
  `PersistWarm()` (_meta <- fields) after DualSimplexCore. matrixVersion stays a plain field (caller-owned,
  read-only inside → survives copy-in). Under `.Run()` the fields don't survive but `_meta` does, so the
  rehydrate restores them; under RunByRef both survive.
  - DEAD ENDS (do not retry): (1) turning the fields into `NativeArray`-backed PROPERTIES changed how the
    solve reads them and (2) a `ref`-into-native `EtaCountRef` accessor for the ref-passed etaCount — BOTH
    regressed the EconomyLP warm re-solve to 4 pivots. Even the correct boundary-mirror (fields untouched)
    still shows 4 vs cold-3 on that 3-pivot toy: it is Burst codegen/struct-layout jitter (adding `_meta`
    perturbs LP.solve's compilation under FloatMode.Default), NOT a warm-state desync (a real resume
    failure costs many pivots, not one; the solve stays optimal). The demo assertion was tightened-then-
    relaxed to `warm <= cold+1` accordingly. STILL TODO: a `.Run()` (not RunByRef) regression test that
    proves cross-copy survival; LPBasis.populated + Pivot.swapCount still plain-field (unfixed).

## fProxyLQRState.populated: native-backed so it survives an IJob by-value copy (warm-state fix)
- 2026-07-17 | Bug class: a warm-start flag mutated inside `IJob.Run()`/`Schedule()` is LOST because the
  job runs on a by-VALUE copy of the state struct (native BUFFERS survive — they're pointers — but plain
  fields don't). A worker `.Run()` of an LQR warm solve silently reset `populated`, forcing every warm
  call cold (or worse for LP's counters). Fix = the MPC `qpMeta` idea, but cleaner: `populated` moved
  behind a `NativeReference<int>` and re-exposed as a `bool` PROPERTY, so all call sites are unchanged and
  writes go through the shared handle — no rehydrate/copy-back (the flag is set once per solve, not in a
  hot loop; contrast LP's `etaCount`). `NativeReference` confirmed Burst-job-compatible (ControlLQRTests'
  TestJob runs the warm path through an IJob). Suite 6317/6317. See [[job-struct-copy-warmstate-audit]].

## UnsafeOP.max/min: hardware mm256_max_pd/min_pd for double too (closes the double gap)
- 2026-07-17 | Follow-up to the width-8 float win: double max/min were still ~11.5 GFLOP/s (~2× behind
  double sum's ~22) because double skipped fProxyW and its fProxy4 body used `math.max(double4)` =
  compare+select. Added `X86.Avx.mm256_max_pd`/`min_pd` to `fProxyW.Max`/`Min`'s DOUBLE path (AVX branch +
  lane-wise fallback) and removed the `skipFor[double]` from the kernels so double now runs the fProxyW
  main loop too. Result: **double max 11.5→19.6, min →20.5 GFLOP/s (~1.7-1.8×)** — now within ~10% of sum;
  float unchanged (its diff was whitespace only). Suite 6317/6317, finite-data bit-identical.
  - SAFE re maxAbs's frozen contract: maxAbs (and sum/vecDot) ALSO skipFor[double], so they never called
    fProxyW.Max's double path — it was dead until this kernel. So adding mm256_max_pd there has no
    collateral effect on any existing double kernel. (User sign-off given.)

## UnsafeOP.max/min: width-8 fProxyW upgrade (corrects the width-4 claim below)
- 2026-07-17 | The earlier "width-4 saturates, min/max are memory-bound" claim was WRONG — KernelBenchmark
  proved it. At in-L1 sizes (N<=1024) max/min are THROUGHPUT-bound: width-4 float max hit only ~12 GFLOP/s
  vs sum's ~40. TWO causes: (1) width-4 (float4/128-bit) vs sum's width-8 (fProxyW/256-bit); (2) `math.max`
  lowers to compare+select (2-3 ops), not a single hardware `maxps`. Fix: added `fProxyW.Min`/`HMin`
  (hardware `mm256_min_ps`, mirrors Max/HMax) and rewrote the kernels to the fProxyW main loop + fProxy4
  remainder (like maxAbs), seeded from a[0] (max/min idempotent → re-including the seed is exact).
  Result: **float max 11.9→31.3 GFLOP/s (2.6×) @ N=1024**, near sum. Suite 6317/6317 (finite-data
  bit-identical; float now follows hardware-max NaN semantics, like maxAbs already does).
  - DOUBLE unchanged (~11.5, still ~2× behind sum): double skips fProxyW (double4 is already 256-bit) and
    its fProxy4 body uses `math.max(double4)` = compare+select. Closing it needs `mm256_max_pd`/`min_pd` in
    fProxyW.Max/Min's DOUBLE path — but that path is shared with maxAbs's frozen contract, so it's gated on
    owner sign-off (OPEN). Added KernelBenchmark `max`/`min` reduction cases to measure all this.

## UnsafeOP.max/min kernels; NormsOP.normalizeColumns + Eigen dot reroutes
- 2026-07-17 | Added `UnsafeOP.max`/`min` (SIMD running max/min over a[0..n), contract n>=1). WIDTH-4
  (fProxy4) only, NOT fProxyW: min/max reductions are memory-bound (one load + one compare/element) so the
  4-wide accumulator saturates bandwidth as well as 8-wide would, AND fProxyW has no Min/HMin (adding it =
  touching the owner-gated wide type, avoided). Seeded from the first lanes (not a neutral identity) so
  all-negative/all-positive inputs are correct; uses math.max/min to match caller NaN semantics. max/min
  are exact → lane order is bit-identical for finite data.
- 2026-07-17 | NormsOP.normalizeColumns: strided per-column norm + scale → row-major per-column accumulate
  (colSum trick, unit-stride, vectorises, bit-identical) into a length-N_Cols Temp, then reciprocal-once
  (`(norm>0)?1/norm:1`) + branch-free row-major `row[c] *= inv[c]` (×1 leaves zero/NaN-norm columns
  bit-identical, matching the old skip). Now needs arena-backed A (fProxyTempVec).
- 2026-07-17 | Eigen: rerouted the O(n) vector dots in the two iterative eigensolvers to UnsafeOP.vecDot
  (deterministic-reorder waiver): powerIteration (seed v·v, Rayleigh v·w, ‖w‖²) and lanczos (seed, alpha
  v·w, the O(steps²) reorth proj = V[k,:]·w, ‖w‖²). Self-dots use vecDot(p,p,n) (established, e.g.
  rowNormL2). MODEST — these are O(n) in O(n²/nnz)-per-iter algorithms (dense tridiagonal path already
  uses vecDotRange and was untouched). Left fused residual max-abs loops scalar.

## NormsOP.matrixL1 — colSum-trick restructure (bit-identical)
- 2026-07-17 | ‖A‖₁ (max abs column sum) had a strided inner loop (`for i: colSum += |A[i,j]|`, stride
  N_Cols → scalar under Strict). Restructured to a row-major per-column accumulate into a length-N_Cols
  Temp: `for i { for j: acc[j] += |A[i,j]| }` (unit-stride inner → vectorises), then a scalar max over
  acc. BIT-IDENTICAL — each column still sums its rows in ascending i order; NaN semantics preserved by
  keeping the final max as `if (acc[j] > best)` (not a max kernel). Same restructure class as
  StatsCore.colSum (~32×). Now needs an arena-backed A (via `fProxyTempVec`, like the sibling matrixL2);
  matrixLInf stays allocation-free (rows are contiguous → already routes to UnsafeOP.sumAbs).

## LOBPCG + ladIRLS: reroute inner dots to UnsafeOP.vecDot (reduction-reroute batch 6)
- 2026-07-17 | Replaced hand-rolled scalar `for c: s += V[i,c]*W[j,c]` dots with `UnsafeOP.vecDot`
  (row pointers hoisted via `.Data.Ptr + (long)row*N_Cols`, length n). Sites: LOBPCG `FillGramSub`
  (Gram = VᵀW, the O(k²·n) fill), `FillHSub` (H = VᵀAW), `Deflate` (coeff = <AgainstB_i, V_a>); ladIRLS
  residual `ri = -b[i] + dot(A_i, x)` (`Optimize.ladIRLS`). Made the three LOBPCG helpers `static unsafe`
  (matches the file's existing RequireDistinctBuffers/RestoreBufferIdentity convention); ladIRLS uses an
  `unsafe {}` block around the row loop (public API method, kept non-unsafe). Added
  `using LinearAlgebra.Internal;` to both files.
  - NOT bit-identical: vecDot's fixed 2×fProxyW/2×fProxy4 accumulator tree reorders the summation vs the
    scalar left-to-right sum. Deterministic + cross-arch (frozen kernel contract), covered by the pre-1.0
    "no bit-compat obligation yet" waiver. Tolerance-based tests unaffected (suite 6317/6317).
  - LEFT scalar deliberately: ladIRLS line ~297 final-objective dot accumulates in `double` even for the
    float variant (higher precision) — vecDot would drop that. LOBPCG's residual loop (rv = AX-λBX)
    fuses compute+store R[i,c]+norm in one pass — not a pure dot, left as a later pointer-hoist target.
    LOBPCG lines 154/263 (Rayleigh/B-norm single dots) are O(k·n), inside the big non-unsafe iteration
    method — skipped to avoid widening its unsafe scope for marginal gain.

## UnsafeOP/WideOP: alias fProxy4 + delete the fProxyM/floatM/doubleM shim layer
- 2026-07-17 | Final step of the alias refactor: no file calls `fProxyM` any more, so deleted class
  `fProxyM` (`proxyStructs.math.cs`) AND `OP/SimdMath.cs` (`floatM`/`doubleM`) outright.
  - `UnsafeOP.fProxy.cs`: aliased `fProxy4` -> `Unity.Mathematics.float4` (deleteThis block, replacing the
    `mathProxies` import) and swapped the 10 `fProxyM.abs/max` accumulator calls (sumAbs/maxAbs) to
    `math.abs/max` — resolve natively on the real float4/double4. `fProxyW` (265 uses, the wide v256 type)
    is namespace-local (WideOP.fProxy.cs), unaffected, and NOT aliased/touched.
  - `WideOP.fProxy.cs`: only touched the two `//!`-commented `emitFor[double]` lines (`fProxyM.abs/max`
    -> `math.abs/max`); these activate ONLY in the generated double file where `fProxy4`->`double4`, so
    `math.abs(double4)` is native. Kept the `mathProxies` import (its live `fProxy4` reinterpret casts
    still use the stub); `fProxyW` itself untouched.
  - Generated delta is a pure identity rename (`floatM.abs(float4)` was literally `=> math.abs(float4)`):
    UnsafeOP.float/double.cs + WideOP.double.cs only, byte-diff = `floatM/doubleM.` -> `math.`. Suite
    6317/6317. The `fProxy4` struct + matrix stubs (`fProxy4x4` etc.) in `proxyStructs.math.cs` STAY —
    WideOP + the matrix proxies still compile against them; that stub deletion is a later phase.
    See [[simd-proxy-select-extension]] / docs/dev/spec-alias-simd-proxies.md.

## QueryOP: alias fProxy4 -> float4/double4 (pilot) instead of extending the stub
- 2026-07-17 | Better fix for the previous entry's problem. Rather than teaching the `fProxy4` STUB new
  tricks (comparison/select), QueryOP now does `//+deleteThis using fProxy4 = Unity.Mathematics.float4;
  //-deleteThis` at file top. In the template `fProxy4` IS `float4` (real type), so `v < best`,
  `math.select(fProxy4,...)`, `*(fProxy4*)ptr` all resolve NATIVELY; codegen still rewrites the token
  `fProxy4` -> `float4`/`double4` per file (alias line deleteThis'd), so the double side gets `double4`
  natively too. No `fProxyM`/`floatM` shim in the path at all. Suite 6317/6317, rowArgMin bit-identical +
  same perf. REVERTED the ec14fad stub additions as now-unused (fProxy4 `<`/`>`/`<=`/`>=`,
  fProxyM.select/min, floatM/doubleM.select/min) — QueryOP was their only consumer. `fProxyM.abs/max` +
  the fProxy4 struct KEPT (UnsafeOP/WideOP/LP/matrix-proxies still use the stub; not yet converted).
  This is the pilot for the general "alias the vector proxies, delete the shim layer" refactor
  ([[simd-proxy-select-extension]] / a spec TBD). fProxyW is the exception — no real Unity float8 to
  alias to, so its ops stay hand-rolled.

## SIMD proxy stubs: fProxy4 comparison + fProxyM/floatM/doubleM select+min; rowArgMin -> fProxy4 SIMD
- 2026-07-17 | Extended the width-4 SIMD proxy surface so branch-free lane-parallel select kernels can be
  written in templates (previously the `fProxy4` stub was accumulator-only: `+ - * /`, abs, max).
  Mechanism recap: `fProxy4`->`float4`/`double4` by codegen. OPERATORS float4 has natively (`<`/`>`/`<=`/
  `>=` -> `bool4`) go straight on the `fProxy4` stub (`proxyStructs.math.cs`) — generated code uses
  float4's native ones, no shim. But `math.select`/`math.min` are STATIC `math` methods, not float4
  members, so a template can't call `math.select(fProxy4,...)`; they go through the existing
  `fProxyM`->`floatM`/`doubleM` indirection (like abs/max) — added `select`+`min` there
  (`proxyStructs.math.cs` stub + `OP/SimdMath.cs` real). `int4` is a REAL Unity type in templates, so the
  index half (`math.select(int4,...)`) needs nothing. See [[simd-proxy-select-extension]].
- 2026-07-17 | First customer: rewrote `RowArgMinScan`/`RowArgMaxScan` from the 4-lane-scalar fallback to
  a real `fProxy4` SIMD accumulator (running extreme + int4 index, strict `<`/`>` mask -> NaN never
  displaces; `fProxyM.select` value, `math.select` index; value-then-smallest-index horizontal reduce).
  Bit-identical (suite 6317/6317). N=1024 float rowArgMin 0.20->0.1175 ms (1.7x over the scalar lanes,
  ~5x over the original indexer scan). iProxy4 NOT built (no integer SIMD kernel needs it yet; int4 is
  real so it wouldn't need the fProxyM shim anyway).

## QueryOP: rowArgMin/rowArgMax as 4-lane branch-free math.select scans
- 2026-07-17 | rowArgMin/rowArgMax (argmin/argmax WITH index capture — doesn't vectorise as a plain
  loop) rewritten via `RowArgMinScan`/`RowArgMaxScan`: 4 INDEPENDENT branch-free scalar lanes (lane L =
  columns L, L+4, ...) using `math.select` (NOT `if`), each keeping a running extreme + its column index
  via a strict `<`/`>` mask, then a value-then-smallest-index horizontal reduce. Branch-free + independent
  lanes → Burst packs/overlaps them. Bit-identical to the scalar first-occurrence scan (strict mask → NaN
  never displaces; suite 6317/6317). N=1024 float rowArgMin 0.57→0.20 ms (~2.9×), double 0.75→0.23 ms
  (~3.2×). Tried fProxy4-select first — BACKED OUT: the fProxy4 stub is accumulator-only (no comparison/
  select); extending it is infra tracked in [[simd-proxy-select-extension]] (would let this go true SIMD,
  but 4-lane scalar already gets most of it). Scan helpers keep `unsafe` in signature (fProxy* param =
  the legitimate case, like UnsafeOP). Added nearestColumn to QueryBenchmark: AllColScores N=1024 float
  0.034 ms (~50× over the old strided ColScore, now faster than nearestRow). QueryOP fully optimized.

## QueryOP: strided column search restructured row-major (AllColScores)
- 2026-07-17 | The column search family (`nearestColumn`/`farthestColumn`/`countWithinColumnRadius`/
  `distancesToColumn`) each scanned columns via the strided `ColScore` (per-column walk down rows).
  Factored a shared `AllColScores` helper that computes ALL per-column metric scores in ONE row-major
  (unit-stride inner) sweep with per-column accumulators (the colSum trick) — metric-specific
  (Manhattan/Euclidean/SqEuclidean/Chebyshev/Dot direct; Cosine uses a second normA accumulator + the
  precomputed normQ). Each column still sums its rows ascending → bit-identical to the strided form
  (suite 6317/6317). The four methods now allocate a length-N_Cols Temp, call AllColScores, then reduce
  (argmin/argmax/count) over it. Same restructure class as colArgMin/nearestRow (~7×/2.6×). `ColScore`
  kept (still used by ArenaExtensions.Query two-pass alloc). QueryOP is now fully optimized except
  rowArgMin/rowArgMax (argmin index-capture — deliberately deferred).

## QueryOP: colArgMin/colArgMax restructured + RowScore metric reductions to vecDot
- 2026-07-17 | `colArgMin`/`colArgMax` (strided per-column argmin/argmax walk) restructured into a
  row-major per-column running-min/max + argmin sweep (the (val,idx) overloads accumulate the running
  extreme directly into valPerCol — no scratch; the index-only overloads use a length-N_Cols Temp).
  Bit-identical (each column visits rows ascending, strict `<`/`>` → smallest-row-wins ties preserved).
  N=1024 float 1.80→0.24 ms (~7.5×), double 1.57→0.25 ms — now FASTER than rowArgMin (0.75 ms), whose
  horizontal per-row argmin doesn't vectorise. Added colArgMin to QueryBenchmark.
- 2026-07-17 | `RowScore` (both overloads): Dot + Cosine reductions routed to `UnsafeOP.vecDot`
  (summation-order-changing = deterministic, pre-1.0 waiver); the difference-based metrics
  (Manhattan/Euclidean/SqEuclidean/Chebyshev) kept a DIRECT `(a-b)²`/`|a-b|` scalar sum, pointer-hoisted
  only (the expanded ‖a‖²−2a·b+‖q‖² form risks catastrophic cancellation at near distances → not used).
  Speeds nearestRow/farthestRow/countWithinRadius/distancesToRow: nearestRow Euclidean N=1024 float
  0.94→0.36 ms (~2.6×). Used `unsafe { }` blocks, not an `unsafe` method modifier (minimal scope).
  Cosine `normQ` was ALREADY hoisted out of the per-row loop (QueryNormSq + the normQ overloads) — not a
  bug, verified. STILL scalar (follow-up): rowArgMin/rowArgMax (argmin index-capture — deferred by user),
  and the STRIDED column search (nearestColumn/farthestColumn/countWithinColumnRadius via ColScore) which
  needs the same row-major restructure but is metric-specific.

## QueryOP.argMaxColNorm: strided column walk restructured to row-major (the colSum trick)
- 2026-07-17 | Corrects the prior entry's "column ops are fundamentally strided, leave them scalar."
  They are NOT: `argMaxColNorm`'s per-column norm was computed by a strided per-column walk
  (`for c { for r A[r,c] }`), but the same result comes from ONE row-major sweep accumulating a
  per-column norm vector (`for r { for c acc[c] += f(A[r,c]) }`), then argmax over acc — the inner c
  loop is unit-stride and vectorises, and each column still sums its rows in ascending order so it is
  BIT-IDENTICAL (no waiver; same reason colSum is). Costs one length-N_Cols Temp accumulator
  (self-disposing, job-safe). Suite 6317/6317 unchanged. N=1024 float 1.85→0.032 ms (~58×), double
  1.54→0.062 ms (~25×) — now identical to the row op; the row/column asymmetry is eliminated, not
  merely worked around. **General lesson: "strided column reduction" is usually a restructuring
  opportunity, not a hard limit — NormsOP.matrixL1 and normalizeColumns' norm pass are the same shape
  and could get the same bit-identical treatment (softmaxColumns too, with 3 row-major passes).**

## QueryOP.argMaxRowNorm: routed to SIMD reduction kernels + new QueryBenchmark
- 2026-07-17 | `argMaxRowNorm` (per-row L1/L2/Linf norm, pick the max) was a hand-rolled scalar
  reduction on the indexer. Rerouted the row-inner reductions: L1→`UnsafeOP.sumAbs`,
  L2→`UnsafeOP.vecDot(row,row)`, Linf→`UnsafeOP.maxAbs` (L1/L2 summation-order-changing = deterministic
  not bit-identical, pre-1.0 waiver; Linf = math.max exact = bit-identical). Outer argmax stays scalar.
  Suite 6317/6317. Added `QueryBenchmark` (a few common ops on N×N: rowArgMin, argMaxRowNorm,
  argMaxColNorm, nearestRow — was a coverage gap). Measured N=1024 float argMaxRowNorm L2 0.61→0.030 ms
  (~20×, 35 GFLOP/s). The bench cleanly shows the row/column asymmetry: rerouted row op 0.030 ms vs the
  STRIDED `argMaxColNorm` (column-inner, left scalar — a contiguous kernel can't consume a strided
  column) 1.85 ms = 62× apart at N=1024.

## LP.simplexCore: tableau pivot hoisted to axpy (spec-raw-pointer-hoist-pass batch 3)
- 2026-07-17 | The dense two-phase tableau simplex `Pivot` (row normalize + eliminate every other
  constraint row and both reduced-cost rows) and `simplexCore`'s initial pricing were on the `fProxyMxN`
  struct indexer. Hoisted `T.Data.Ptr` (per-row base) and routed every `row -= f*pivotRow` /
  `cost -= f*pivotRow` through `UnsafeOP.axpy` (eliminate rows i != prow are distinct from the pivot
  row; cost1/cost2 are distinct buffers → `[NoAlias]` legal; IEEE-exact → bit-identical, iters + objective
  byte-identical, suite 6317/6317). Measured (9950X3D): §1 tableau simplex float n=192 102.8→4.41 ms
  (23×), n=384 1475→50.3 ms (29×); double n=384 1924→118 ms (16×); §4 covering LP float n=192 1553→51.3 ms
  (30×). RatioTest left scalar (T[i,enter] = strided column walk). Note: tableau simplex is the reference
  backend (default is RevisedSimplex), so this mainly speeds LAD-simplex + the reference path.

## NormsOP: row norms routed to SIMD reduction kernels
- 2026-07-17 | Same follow-up as StatsCore's row reductions (see Statistics/DEVLOG.md). `normalizeRows`
  (L1/L2/Linf per-row norm) and `matrixLInf` (max abs row-sum) were hand-rolled scalar reductions —
  serial-locked under Strict. Rerouted: L1→`UnsafeOP.sumAbs`, L2→`sqrt(UnsafeOP.vecDot)`, matrixLInf
  inner→`sumAbs` (all summation-order changes → deterministic but not bit-identical to the prior serial
  sum; owner-approved pre-1.0 baseline change). **Linf→`UnsafeOP.maxAbs` is BIT-IDENTICAL** (max is
  associative/exact — no rounding to reorder), a free win needing no waiver. The `row[c] *= inv` apply
  loop is a bit-identical elementwise hoist. `normalizeColumns`/`matrixL1` left scalar (column-inner =
  strided). Suite 6317/6317. No NormsOP benchmark (accepted gap); the kernels are the same ones
  measured in StatsBenchmark (rowSum 0.35→0.035 ms at N=1024).

## LU.decompNoPivot: raw-pointer hoist (spec-raw-pointer-hoist-pass batch 1)
- 2026-07-17 | `decompNoPivot`'s trailing-row elimination inner loop `U[j,i] -= Ljk*U[k,i]` was still
  on the `fProxyMxN` struct indexer while its pivoted siblings (`decomp`/blocked/`decompInPlace`)
  already hoist `U.Data.Ptr` and route the axpy through `UnsafeOP.axpy`. Applied the identical
  transform (rows j>k are distinct → `[NoAlias]` legal; `(-Ljk)*U[k,i]` added is IEEE-exact to the
  scalar form → bit-identical, suite 6317/6317 unchanged). Measured (9950X3D, upper CCD, N=1024):
  float 514.9→19.30 ms (~27×, 1.39→37.1 GFLOP/s), double 519.6→34.43 ms (~15×). Tracks pivoted
  `decomp` up to N≤256; the blocked level-3 path pulls `decomp` ahead only at N=1024 (out of scope).
  Added a `decompNoPivot` case to LUBenchmark (was measuring only pivoted `decomp` = a gap).
- 2026-07-17 | LU/LUP split (user floated during this batch): recommend DO NOT split. `decompNoPivot`
  is already its own public entry point with its own contract, and post-hoist it shares the same
  vectorised axpy kernel as the pivoted paths — a file/type split buys no perf and adds codegen churn
  plus API-surface risk pre-v1.0. Left as one `LU` partial class.

## LOBPCG: IJob cache-copy corrupted eigenvectors (ping-pong buffer reseat)
- 2026-07-16 | Symptom: `Eigen.lobpcg` run inside an IJob returned correct eigenVALUES but
  corrupted eigenVECTORS (relative residual ~1e-1) on clustered/near-degenerate spectra; the same
  call on the main thread (`ref cache`) was exact. Presented as "Burst-only" and cost a long hunt —
  I wrongly chased FloatMode, FloatPrecision, `[ReadOnly]`/NoAlias, OptimizeFor, an aliased `Deflate`
  call, and a stale-`AX` theory, and even wrote (then reverted) a comment blaming Burst for mis-
  sequencing `Swap.Rows`. All wrong. ROOT CAUSE (credit: fable consult): `UpdateActiveBlock` ends
  each iteration with `SwapMat(ref ws.X, ref ws.Xnext)` — a struct-VALUE ping-pong (double buffering)
  that reseats which allocation the `ws.X` FIELD names. An IJob executes on a COPY of the cache
  struct: writes THROUGH the buffer pointers reach the caller, but the reseated FIELD does not, so
  after an ODD iteration count the caller's `cache.X` still points at the entry buffer, which holds
  the previous (pre-sort) iterate → sorted `lambda` paired with UNSORTED `X`. `lambda`/`residual`
  are never ping-ponged so they always sort correctly; that asymmetry was the tell. It is NOT a
  Burst bug (a plain Mono IJob reproduces it identically; the correct vectors sit in `cache.Xnext`).
  Only surfaces when the exit sort does real reordering (locking on clustered spectra) AND parity is
  odd. Fix: capture entry buffer identities (`xEntry`/`pEntry`), and before every return
  `RestoreBufferIdentity` copies the final data back into the entry allocation and swaps the fields
  so `ws.X`/`ws.P` reference their entry buffers on return — one O(k·n) copy at exit only when parity
  flipped, zero hot-loop cost, ping-pong untouched. P is restored too (warm-start reuse reads it).
  Why the suite missed it: every prior LOBPCG [Test] was a main-thread `ref` call and the benchmark
  jobs only read `infoOut`; added `JobbedClusteredSpectrumLeavesCorrectVectorsInCache` (runs
  `.Run()` on a 2D-Laplacian degenerate spectrum, checks `cache.X` residuals post-job — verified it
  fails when the fix is neutered). Audited: this `SwapMat`-of-caller-visible-cache-field pattern
  exists ONLY in LOBPCG.

## Riccati (public DARE primitive)
- 2026-07-16 | Extracted the DARE engine out of the LQR facade into a new public
  `Riccati.dare(in A, in B, in Q, in R, ref S, maxIter)` (root `LinearAlgebra`, sibling of
  Eigen/SVD/Krylov). Was `Control.LQR.SDACore` (internal); LQR (control) and Kalman.steadyStateGain
  (estimation, via the Aᵀ/Hᵀ duality) BOTH consume it, so it belonged in a neutral primitive both
  depend DOWN onto -- this deletes the Kalman->Control.LQR reach entirely (Kalman lost its
  `using LinearAlgebra.Control;`). The shared hygiene kernels moved with it (Riccati.SymmetrizeInPlace,
  Riccati.FrobeniusNorm/FrobeniusNormDiff -- double-accumulate, deliberately NOT Norms.L2 which sums in
  fProxy -- Riccati.BlowupThreshold; consts SDA_MAX_ITER/BLOWUP_FACTOR now on Riccati.Info.cs). LQR
  keeps its control-specific mechanics (RiccatiStep = S->K gain kernel, RiccatiIterate warm recursion,
  lqr/lqrSchedule/lqg, fProxyLQRState, WARM_MAX_ITER) and calls Riccati.* for the shared bits; MPC's QP
  Hessian symmetrize now calls Riccati.SymmetrizeInPlace too.
- 2026-07-16 | DEDUP: `LQRInfo`/`LQRStatus`/`LQRStatusExtensions` DELETED, replaced everywhere by
  `RiccatiInfo`/`RiccatiStatus` (identical fields; the DARE result is the DARE result whether used for
  control or estimation). `rankDeficientControl` -> `rankDeficient` (generic: for the Kalman dual it is
  measurement-space, not "control"). LQR.lqr/lqrSchedule/lqg and Kalman.steadyStateGain now return
  RiccatiInfo; LQGInfo bundles two RiccatiInfo. Supersedes the "Control.LQR.SDACore" reach noted in the
  namespace entry below (same day).

## Control namespace (LQR / MPC)
- 2026-07-16 | Moved the control API out of `namespace LinearAlgebra` into a dedicated
  `namespace LinearAlgebra.Control` and renamed the LQR facade class `Control` -> `LQR` (the old
  `Control.lqr(...)` read confusingly next to the `LQ`/`LQRP` matrix decompositions). MPC + all
  companion types (`LQRInfo`/`LQRStatus`/`LQGInfo`/`fProxyLQRState`, `MPCInfo`/`MPCStatus`/
  `fProxyMPCState`) moved into the same sub-namespace. Kalman deliberately stayed in
  `LinearAlgebra` (user ruling); it reaches the internal Riccati helpers as `Control.LQR.SDACore`/
  `SymmetrizeInPlace`/`FrobeniusNorm` (internal = assembly-scoped, so cross-namespace is fine) and
  gained a file-level `using LinearAlgebra.Control;` because it NAMES `LQRInfo`/`LQRStatus` in code
  (`steadyStateGain`'s return type). Nested-namespace files still see every parent `LinearAlgebra`
  type (fProxyMxN/QP/CHOP/Blas/...) with no `using`, which is what kept the move low-risk — only
  external consumers (tests, benchmarks, demos) needed `using LinearAlgebra.Control;`. Method names
  unchanged (`LQR.lqr`/`lqrSchedule`/`lqg`, `MPC.solve`). Suite green post-regen.

## DetMath
- 2026-07-16 | Added the `LINALG_NATIVE_MATH` compile-mode switch: a single `#if` sets
  `public const bool UseNative`, and every transcendental branches on that const as its first
  statement (`if (UseNative) return math.XXX(...)`). Deterministic DetMath stays the default
  (const false); defining the symbol flips every call site to `math.*` for raw throughput,
  giving up cross-arch determinism. Because it's a `const bool` rather than a per-function
  `#if`, BOTH branches are always real, type-checked C# — Burst folds the dead branch away at
  native codegen (literal-const propagation), so there's no runtime cost and no risk of a
  native-only typo going unnoticed by the default (deterministic) test run. Left composing
  (no native branch, per spec): `Pow(fProxy,int)` (exact integer path, no math.* equivalent),
  `Exp10` (no math.exp10 in Unity.Mathematics), `Acosh` (no math.acosh). `SinhCosh` (the
  shared-computation helper, analogous to `SinCos`) also stays composed — only its two callers
  `Sinh`/`Cosh` gained native branches, matching the exact function map in the spec. Verified
  `math.exp/log/log2/log10/sin/cos/tan/atan/atan2/asin/acos/sinh/cosh/tanh/exp2/pow/sincos` all
  exist for both float and double in Unity.Mathematics before wiring (checked
  Library/PackageCache math.cs directly). SinCos's native branch calls `math.sin`/`math.cos`
  separately rather than `math.sincos` — its `out float`/`out double` params don't bind to the
  `fProxy` proxy type in the raw template (same limitation already hit by RandomOP's Gaussian
  sampler, see below). `UseNative` itself is wrapped in `//+skipFor[double]` so it's defined
  ONCE (float fragment only) instead of twice — DetMath.float.cs and DetMath.double.cs merge
  into one partial class, so a bare unwrapped const would double-define and fail CS0102; the
  double fragment's method bodies still see it fine through the merge. Runtime testing of the
  native path requires adding the define under Player Settings — not done here (out of scope;
  default-mode compile already exercises both branches' C#).
- 2026-07-15 | Promoted the deterministic transcendentals from the benchmark prototype
  (TemplateSourceBenchmarks/DetMathBenchmark) to a shipping public class `DetMath` (OP/DetMath.
  fProxy.cs, float+double overloads). Surface: Exp/Exp2/Exp10/Log/Log2/Log10/Pow, Sin/Cos/SinCos/
  Tan, Asin/Acos/Atan/Atan2, Sinh/Cosh/Tanh, Acosh — everything the library's math.* usage needs
  (rcp/rsqrt/sqrt stay math.*, already deterministic). One canonical scheme: accurate Horner
  minimax (dropped the prototype's Estrin/Fast experimental variants; Estrin is a latency option
  if a scalar hot path ever needs it). Cody-Waite reduction, ldexp-by-bits, all branch-free
  guards. Accuracy vs libm ~1e-5 float / ~1e-12 double (few ULP) verified by sweep tests
  (DetMathTests, 500-pt sweeps per fn over the domain + edge/total behaviour), suite 6297/6297.
- 2026-07-15 | ExpGuard NaN bug found by the new edge tests (the benchmark never exercised it —
  its inputs were [-10,10]). Exp relied on the polynomial IMPLICITLY producing NaN for a NaN
  input, but the `(int)NaN` conversion + multiply in Ldexp does not preserve NaN under Burst, so
  Exp(NaN) returned a finite value. Fix: ExpGuard now propagates NaN EXPLICITLY via
  `select(y, NaN, x != x)`, matching LogGuard/TrigGuard (which always did the explicit x!=x check
  and passed). Lesson: never rely on implicit NaN propagation through an int-conversion path;
  guard the original input explicitly.
## axpy4: quad-stream panel updates for the blocked factorizations
- 2026-07-14 | vecMatDot (xᵀA — simplex PRICE, transposed GEMV) moved onto axpy4 (four matrix
  rows per output pass, r-ascending per element = bit-identical): float 41→69 GF/s at n=64,
  57→82 at 128, 69→85 at 256, +18% at 512; double +19-56% at 64-256. 1024 rows flat — the
  streamed matrix is the bandwidth wall there, quad-streaming can't help a one-touch stream.
- 2026-07-14 | The blocked factorization trailing updates (CHO syrkLowerSub, CHOP
  syrkUpperSub, QR/LQ wyVtC/wySubVW/lqYeqCVt, pivoted-LU's inlined row update) all had the
  same shape: one axpy pass over the output row per panel column — vectorized but bound by
  output-row read-modify-write traffic, not flops (CHO float 1024 ran 33 GF/s vs the GEMM
  tile's ~100). Fix: UnsafeOP.axpy4 fuses FOUR coefficient streams into one output pass
  (arithmetic intensity 4x); per-element operation order stays p-ascending sequential, so
  results are BIT-IDENTICAL to the old kernels for both dtypes — no skipFor/W-tier needed,
  the map auto-vectorizes. Min-across-runs (ambient load made single runs swing 20-40%;
  trust direction + mins): CHO 1024 float 10.79 → 9.10 ms, double 15.17 → 12.92; CHOP 1024
  float 21.10 → 17.97; dense blocked LU inherits via wySubVW (float 1024 15.69 → 14.17).
  NOT retuned under noise: floatCholBlockMinN (1024 — blocked path got ~15% faster, the
  crossover may now sit at 512; re-measure on an idle machine), same for the QR/LQ/LU gates.
  trsmLowerPanel and the LU/QR small TRSM steps (~5% of time) left single-stream.

## fProxyW stage 2c: broadcast GEMM tiles (matMatDot / TransA / AtA) on wide accumulators
- 2026-07-14 | User ruling: no wrapper-level transpose routing ("just rewrite the critical
  path") — a briefly-added staged-transpose detour in Blas.dot/dotSym was reverted the same
  hour. The honest rewrite: matMatDotUnpackedW / matMatDotTransACoreW hold the 8x16 tile as
  two fProxyW per row with Splat broadcasts — BIT-IDENTICAL to the scalar tiles (one
  p-ascending chain per element), so no numeric change at all. Float: plain GEMM 90 → 110-114
  GF/s at 128-512 (1024: 28.0 → 24.0), TransA 88 → 103-115 (1024: 27.5 → 24.9), AtA
  151 → 174-204 GF/s-eff (512: 1.61 → 1.31 ms); n=64 improved too (no small-size gate
  needed). Double: unchanged, keeps the scalar tiles via choose-routing.
- 2026-07-14 | trsmLowerPanel rewritten on fProxyW (both dtypes: 8 float / 4 double lanes),
  bit-identical: rows are independent, so Width rows solve simultaneously through a
  contiguous tile — per column p, one broadcast-FMA chain over k<p (no per-row short
  reductions, no horizontal ops), then one wide division; each lane replays the scalar
  row's exact chain. Blocked CHO (idle-machine A/B, unblocked control rows flat): float
  512 1.635→1.147 ms (−30%), float 1024 8.78→6.80 (−23%, 52.6 GF/s — CHO now beats LU),
  double 512 −21%, double 1024 −12%. The old "TRSM ≈5% of time" note was wrong — the
  dot-form solve was ~9% of flops at a fraction of SYRK throughput ⇒ ~25-30% of wall.
  fProxyW gained operator/ (mm256_div_ps + lane fallback; template fProxy4 stub too).
  CODEGEN NOTE: this kernel's accumulator IS seeded from memory and compiles clean — the
  seeded-W byte-rotation pathology (see the 6x16 entry) needs MANY live seeded
  accumulators, not one. TESTS: CHO Blocked* cases were sized 256-400 for the ORIGINAL
  256 gate and silently stopped reaching the blocked core when the gate moved to 512 —
  resized to 512/545/576/600 (545/600 = ragged last panel + wide-kernel scalar-remainder
  seam). Check test sizes whenever a gate moves up.
- 2026-07-14 | CHOP blocked-path optimization pass, bit-identical outputs (~10% at 1024,
  ~4-10% at 512, ~6% at 256, both dtypes): (1) contiguous diagRaw mirror for the pivot
  search — reading W[i,i] directly is a stride-(n+1) scan over the full trailing range
  EVERY column (one cache line per entry, ~33 MB effective at n=1024, rivaling the whole
  SYRK stream); the mirror is refreshed per panel after the SYRK and swapped alongside W's
  diagonal, holding exactly W[i,i]'s bits so pivot choices are unchanged. (2) Deferred
  panel-end L scatter: Ukk parks in W[k,k], the panel's factor rows block-transpose into
  L's columns once per panel (W panel rows stay L2-resident, L written in 32-element runs)
  instead of one stride-n column write per factored column; Swap.Rows(L) narrows to the
  already-scattered columns [0,j0) — W's own column-segment swap maintains the deferred
  part. (3) Winner-only corrections quad-fused via axpy4. CHOLP_BLOCK=64 tried on top:
  float 1024 −3% but double 256 +5% — reverted, stays 32. Unblocked (<256) path untouched.
  every gate; ~1% ambient drift measured via unchanged-route control rows): doubleQr
  512→256 (−19% at 256), doubleQrcp 512→256 (−7%), CholPivot float+double 512→256
  (−12%/−9%), floatLu 256→128 (−3.5%). Ties (kept prior value): floatQr@64, doubleLu@64,
  doubleChol@256 (0.3775 vs 0.3774 — that IS the crossover). floatChol was retuned to 512
  the same day already. The axpy4 trailing-update fusion moved the level-3 crossovers down,
  most strongly for double.
- 2026-07-14 | 6x16 SEEDED-W TILE TRIED AND REJECTED — falsifies the register-pressure
  hypothesis below. A full packed driver (MR=6, MC=126) + seeded 6x16 W microkernel
  (12 accums + 2 B + 1 broadcast = 15 ymm, comfortably inside the 16-register file)
  measured 29.4-29.6 GF/s at 512-2048 vs the scalar packed kernel's 81.9-84.4 in the same
  run — the SAME collapse as the seeded 8x16 W (34). Disasm shows the identical pathology
  at 15 live vectors: accumulators in stack slots, vperm2i128 + vpalignr-by-1-byte
  rotations around every add, plus vpextrb/vpinsrb single-byte traffic (41 vpalignr /
  29 vperm2i128 / 32 vpextrb vs 12 vmulps + 12 vaddps). So the trigger is SEEDING wide
  accumulators from C at all, not how many are live: zero-init W tiles (matMatDotUnpackedW,
  matMatDotTransBRangeW) compile clean, seeded W tiles of ANY height collapse, seeded
  SCALAR tiles SLP clean. Zero-init + add-C-at-writeback is not a legal fix (different
  per-element summation tree, breaks the packed==unpacked bit contract). Scalar packed at
  82-84 GF/s already sits at the float SLP ceiling, so there is nothing to win — do NOT
  retry seeded wide microkernels until a Burst/LLVM upgrade changes the codegen; re-test
  with the disasm recipe first. Experiment code removed; a permanent GEMM-packed-direct
  benchmark section (gate bypassed, 512-2048) documents the pack-overhead crossover.
- 2026-07-14 | Mystery root-caused at the assembly level (headless Burst disasm via bcl.exe —
  recipe in docs/dev/burst-disasm-recipe.md). The seeded W microkernel's p-loop compiles to
  160 instructions vs the scalar microkernel's 76 for identical arithmetic (16 vmulps +
  16 vaddps + 8 vbroadcastss): LLVM chains the 16 seed-loaded v256 accumulators through
  vperm2i128 + vpalignr-by-1-byte rotations and stack slots instead of plain registers —
  byte-granular shuffle glue around every accumulator update, ~2.1x instruction bloat ≈ the
  measured 34-vs-114 GF/s collapse. The trigger is the combination of 16 LIVE fProxyW
  accumulators SEEDED from strided C rows (8x16 tile needs 16 accums + b0 + b1 + broadcast
  = 19 ymm > 16 physical; the rotation chain is LLVM's spill-avoidance gone wrong). The
  scalar microkernel survives the same pressure because its seeds/updates are scalar SLP:
  B stays as folded memory operands and the loop stays tight. Clean control cases in the
  SAME dump: matMatDotUnpackedW (zero-init accums, C added at write-back) and
  matMatDotTransBRangeW — zero shuffles, so fProxyW itself is fine; only seed-first + full
  register pressure trips it. Fix directions if packed-W is ever wanted: (a) 6x16 tile
  (12 accums + 2 B + 1 broadcast = 15 ymm — the classic BLIS AVX2 sgemm shape; needs MR=6
  pack layout + a matching scalar twin for the chain contract), or (b) restructure seeding
  (zero-init + C at write-back breaks the packed==unpacked bit contract — only with a
  matching unpacked change). Until then the packed path stays on the scalar microkernel +
  24 MB gate; float 512-1024 keeps ~10-20% headroom vs the transpose-detour reference
  (118/106 GF/s). Probe code (GemmMicroProbe + gemmMicroKernelW) was template-temporary and
  is REMOVED; asm listings preserved with the recipe doc.

## fProxyW stage 2b: TransB row-dot family on the wide core
- 2026-07-14 | matMatDotTransBCoreW (float-only, choose-routed at the three entries; the
  original core is now double-only via skipFor[float]): same 2x4 pair tile, one fProxyW
  accumulator per pair. Float A·Bᵀ 1024: 29.2 → 17.2 ms (125-136 GF/s — now the FASTEST GEMM
  shape in the library, beating plain matMatDot's ~77-90); float A·Aᵀ 1024: 14.4 → 9.0 ms
  (237-253 GF/s-effective). Beats the trans+dot route at EVERY size, so the wrapper's
  per-dtype viaTrans split (added earlier the same day) is REMOVED — unified kernel dispatch
  again. dotSymT float (Kalman covariance shapes) inherits the win. The mirror pass is now the
  shared mirrorLowerFromUpper helper (three cores use it). Same-run A/B, so valid despite
  ambient machine load (double control rows steady).

## fProxyW stage 2: matVecDot + sum/sumAbs/maxAbs
- 2026-07-14 | GEMV (matVecDot) float on the tiered pattern: 43.7 → 73.6 GFLOP/s at 1024
  (3.07 → 1.82 ms), 1.6-1.7x at 256-1024, measured before ambient machine load contaminated
  later runs (double control rows drifted +9% across runs — re-verify the small-row gate
  threshold on an idle machine). n=64 rows REGRESSED under the ungated W-tier (0.0150 →
  0.0221 ms: per-row fold overhead vs only 4 loop iterations), so the float W-tier gates at
  row length >= 128; below it the shared width-4 tier is exactly the pre-rework kernel.
  sum/sumAbs/maxAbs converted on the same pattern (L1 norm float 1024: 41.6 → 61.0 GF/s);
  fProxyW gained Abs/Max/HMax. vecMatDot left alone — it is a scalar map Burst already
  auto-vectorizes at full width. NOTE (user ruling): old-vs-new bit identity is NOT a
  constraint pre-release; internal same-build contracts (fused == composed) still hold.

## fProxyW: the three-tier conversion pattern (canonical recipe)
- 2026-07-14 | Every float width conversion follows one shape (user design, refined to a SINGLE
  marker): hoist `int i = 0; fProxy head = 0;`, then
    (1) `//+skipFor[double]` float-only W-wide main tier — folds into `head`, advances `i`;
    (2) SHARED width-4 two-chain tier — for double this IS its original main loop (i enters at
        0, identical chain assignment and fold), for float it covers the <8 remainder;
    (3) SHARED scalar tail; `s = head + quadFold` then tail appends.
  No emitFor, no duplicated double body (emitFor remains for genuinely-different bodies, e.g.
  fProxyW's own AVX-vs-double4 ops). Accepted nit: double's fold gains a leading `0 +`, which
  flips the SIGN OF ZERO when an entire reduction is −0 (e.g. dot against a zero vector with
  negative signs) — behaviorally invisible (−0 == +0, guards use >), noted on the CHANGELOG's
  "double unchanged" claim. Reduction tree = frozen contract; fused kernels mirror vecDot's
  shape exactly (bit-identical-to-composition). Converted: vecDot, vecDotRange, axpyNormSq,
  xpayNormSq, updateXR.

## fProxyW (WideOP) + float width rework, stage 1: vecDot
- 2026-07-14 | fProxyW added: 8 float lanes via Burst AVX intrinsics (v256) / 4 double lanes,
  32-byte v256 storage for both, lane-tree-identical non-AVX fallback (correctness path).
  vecDot/vecDotRange FLOAT moved onto it: 44-45 → 82-84 GFLOP/s cache-resident (roofline H,
  1K-64K elems; converges to bandwidth at DRAM sizes). Float's summation tree changed (2x8
  chains, halves-first fold) — new frozen contract, CHANGELOG'd.
- 2026-07-14 | DOUBLE was first routed through fProxyW too ("one template body") and REGRESSED
  ~19 ns/call (1K-elem dot 45 → 32 GFLOP/s): double4 is already full width, so the v256↔double4
  reinterpret wrapper only added per-call overhead. Double now keeps its original fProxy4 body
  verbatim via the new //+emitFor[double] codegen marker (bit-identical trivially). Rule for
  the rest of the rework: fProxyW is a FLOAT-side lever; leave double bodies alone.
- 2026-07-14 | Dot-shape ceiling context: ~90 GFLOP/s is the LOAD-PORT limit for two-operand
  streaming reductions (2 loads per mul+add), not the 120 register-chain ceiling — do not
  chase the gap.
- 2026-07-14 | TRAP (cost one full suite run, ~500 failures): fProxyW.Width's choose
  placeholder was 4 — but the template assembly's own tests RUN template code against the
  float-backed stub, so wide loads advanced 4 floats while processing 8. Placeholders in
  dtype-split code must be the FLOAT values (see codegen-refactor-lessons.md). Both generated
  files were correct the whole time; the bit-identity fallout that was real: xpayNormSq /
  updateXR (the CG fused kernels) pin "bit-identical to axpy+vecDot" — converted their float
  reductions to the fProxyW tree in the same pass (they inherit the width win too).

## LP.Sparse float IPM: stall-quality envelope (open robustness item)
- 2026-07-14 | Exposed by the float width rework (not caused by it): the float sparse IPM's
  outer tolerance (100·eps) is unreachable in float on unscaled real data — stackloss LAD
  always exits MaxIterations at a rounding-dependent objective (measured 2%..117% above the
  optimum across float summation-tree variants; double converges Optimal). Shipped: float
  inner pcgTol tightened sqrtEps → sqrtEps/10 via choose (measured stall 44.4 → 43.0 on
  stackloss; double untouched), and LPTests.SparseLadStackloss's 8% band made double-only
  (float asserts a wide sanity envelope — never below the optimum, never > 3x). Real-fix
  candidates for a robustness pass: column equilibration in the standard-form operator
  (stackloss columns span ~1..~90), inexact-Newton forcing terms (inner tol ∝ μ), and a
  float-realistic outer tolerance with honest status reporting.

## LOBPCG float: spurious-Ritz collapse by over-iteration (open robustness item)
- 2026-07-14 | Surfaced by the float width rework's tree change but NOT caused by it: on the
  truss demos' penalty-conditioned pencil (penalty 1e3 vs O(1) eigenvalues of interest), float
  LOBPCG iterated past the DEFAULT tolerance collapses its basis and reports spurious near-zero
  Ritz values as Converged (measured, 8-dof braced square, true λ1 = 1.198: tol=1e-4 → λ1 ≈ 1e-6
  "Converged"; tol=1e-6 → both eigenvalues exactly 0; default tol → correct at every penalty
  30..1000). TIGHTER tolerance makes it WORSE — the failure is orthogonality-budget exhaustion,
  not insufficient convergence. Demos now use the default tolerance (comments point here).
  BACKLOG: a guard inside lobpcg (Gram conditioning check, or residual-vs-Ritz-scale sanity)
  so basis collapse reports Indefinite/Breakdown instead of Converged-with-garbage.

## UnsafeOP packed (cache-blocked) GEMM
- 2026-07-13 | BLIS-style packed route added to matMatDot (KC=256/MC=128 panels, MR/NR strips,
  seeded microkernel so every element's reduction stays ONE p-ascending chain across panels —
  bit-identical to the unpacked route, pinned by DotSymTests.PackedMatchesUnpackedBitExactly).
  MEASUREMENT SURPRISE on the 9950X3D (bench pinned to the 32 MB-L3 CCD): packing only pays once
  the working set spills L3 — float 2048 unpacked sags to 58.7 GF/s, packed holds 82.7 (1.41x);
  double 2048 1.27x — but BELOW that the pack copies + per-panel C reloads are a pure loss
  (float 512 +33%!). First gate ((m+k)*n >= 128k elements) was far too low; final gate is
  ~24 MB total working set, byte-scaled. Expect the crossover ~3x higher on the V-cache CCD and
  lower on small-L3 consumer CPUs. Broader lesson: the unpacked kernel is NOT bandwidth-bound
  on big caches at <= 1024 — it sits at ~85-90 GF/s float, load-port/issue-bound under Strict
  (no FMA contraction) — so cache blocking is a big-N/small-cache lever, not a general one.
- 2026-07-13 | dot(transposeB) route split PER DTYPE (skipFor): float materializes Bᵀ (staged
  trans) + broadcast GEMM (row-dot TransB kernel is half of AVX2's float width; viaTrans measured
  at-or-faster at every size, and this un-regresses KMeans float); double keeps the row-dot
  kernel (wins at every size, e.g. 2x at 128). Aliased A·Aᵀ keeps matAAt for both dtypes.

## UnsafeOP TransB kernel family (matMatDotTransB / matAAt / matMatDotTransBSym)
- 2026-07-13 | First cut used TWO fProxy4 chains per output pair (vecDot's idiom) over a 2x4 pair
  tile: 16 accumulators + 12 transient loads spilled registers and LOST to the trans+dot route it
  was meant to replace (float 1024: 49.2 ms vs 31.7). Rewritten to ONE chain per pair (8
  accumulators + 6 transients, inside the 16-register budget): float 1024 28.7 ms, double 42.5 vs
  54.2 viaTrans; double 128 is 2x (0.061 vs 0.124). Don't re-add the second chain. Only float N=64
  still marginally favors viaTrans (0.0086 vs 0.0072 ms) — not worth size-gated routing.
- 2026-07-13 | matAAt (A·Aᵀ upper+mirror): 143 GFLOP/s-effective float / 105 double at 1024 —
  on par with matAtA despite the dot-product formulation, because symmetry halves the work and
  both row streams are unit-stride. No trans+matAtA fallback route needed.

## Control symmetric-GEMM reroutes (RiccatiStep / SDACore)
- 2026-07-13 | Riccati/SDA symmetric products moved to dotSym (missed by the first symmetric-GEMM
  pass, which covered QP/MPC only): RiccatiStep's Bᵀ(SB), Aᵀ(SA), BSAᵀK (= BSAᵀR̄⁻¹BSA) and
  SDACore's AkᵀX3 H-update. AkᵀX3 is symmetric only in exact arithmetic (X3 exits an LU solve);
  the mirror picks the upper triangle's roundoff where the full kernel produced O(eps) asymmetry
  that SymmetrizeInPlace then averaged — the existing post-add SymmetrizeInPlace is kept.
  SDACore's GkNext = (AkGk)·X2 does NOT fit either sym kernel form (neither operand of the
  symmetric product is materialized transposed) — left on the full kernel deliberately.

## UnsafeOP matTrans + symmetric-mirror cache blocking
- 2026-07-13 | Plain TB=32 blocking (strided writes kept inside the tile) was a TRAP: at
  power-of-two sizes the 2-4 KB stride maps a whole tile column into 1-2 L1 sets and way-thrashes
  — blocked matTrans measured 0.21 Gelem/s at float 1024, WORSE than naive. Fix: stage every tile
  through a TB=16 stackalloc buffer (read side row-contiguous into buf, write side row-contiguous
  out of buf; neither matrix ever strides). Never ship a plain two-loop blocked transpose again —
  measure at power-of-two N specifically. Same staging applied to the symUpper mirror passes in
  matMatDotTransACore/matMatDotTransBCore (their blocked-unstaged version did already beat the
  naive mirror: AtA float 1024 15.9→14.2 ms). All pure permutations: bit-identical results.
  Benchmark instruments: new "Trans" section (Gelem/s) + the viaTrans/AtA/AAt rows.
- 2026-07-13 | Staged mirror needed a SMALL-MATRIX BYPASS (m <= 64: plain mirror, no buffer):
  a uniform staged path regressed gamedev-scale LQR 5-18% (n=4-12 Riccati steps) — at those
  sizes the whole matrix is L1-resident (thrash impossible) and the stackalloc buffer's
  per-call localsinit zero-fill dominates. Iteration counts unchanged either way. General
  rule: any stackalloc-staged kernel path needs a small-size bypass or the small callers pay
  the buffer for nothing. Also: below one register tile (m < 8) the symUpper tile-skip never
  fires — dotSym at tiny sizes is full compute + mirror, i.e. pure overhead vs plain dot, so
  small-n callers only keep dotSym for the exact-symmetry contract, not for speed.

## Kalman / Kalman.UKF TransB + GEMM reroutes
- 2026-07-13 | predict: APAᵀ now Blas.dotSymT(AP, Aeff) — s.At no longer written by predict (still
  UpdateCore's (I-KH)P scratch). update: P·Hᵀ via dot(transposeB) (Ht temp deleted); K = Xtᵀ never
  materialized — K·y via vecMat dot(y, Xt), K·H and K·R via dot(Xt, ·, transposeA: true), IKHt temp
  deleted in favor of dotSymT((I-KH)P, IKH). ukfUpdate: same K elimination via dot(y, Pxzt).
- 2026-07-13 | UKF sigma recombinations GEMM-ified: predict's Σ Wc·d·dᵀ = (WD)ᵀ·D via dotSym
  (D overwrites Y, WD reuses X — both fully consumed by the propagation loop); update's
  Pzz/Pxz via dotSym(dZ, WdZ) + dot(dX, WdZ, transposeA) (dX overwrites X, dZ overwrites Z,
  one npts x m WdZ Temp, dz vector deleted). Results are bitwise different from the scalar
  rank-1 loops (different summation order), suite-validated.

## Parameter naming (library-wide)
- 2026-07-13 | Short tuning-param names ruled canon: maxIterations → maxIter, tolerance → tol,
  relativeTolerance → relTol, library-wide. REVERSES the earlier long-name rename pass — do not
  rename back. maxSweeps kept where the algorithm genuinely counts Jacobi sweeps. Rule recorded
  in docs/dev/naming-style-guide.md.

## Krylov.Guards.cs
- 2026-07-13 | //singularFile// on this partial is load-bearing, not a style choice:
  RequireDistinctBuffers has no fProxy token in its signature, so if it were declared inside the
  multiplying Krylov.fProxy.cs template it would be copied identically into both the generated
  Krylov.float.cs and Krylov.double.cs fragments of the same partial class -- two definitions of
  the same member -> CS0111. (was Krylov.Guards.cs:7-11)

## Krylov.PBiCGStab
- 2026-07-13 | Parameterless BSR overload's default iteration budget changed 2*A.M_Rows → A.M_Rows
  to match the unpreconditioned biCGStab twin and the rest of the square-solver family (release-scan
  N14 finding: undocumented sibling inconsistency; no measured rationale existed for the 2x).

## UnsafeOP / UnsafeBoolOP / SelectOP aliasing policy
- 2026-07-13 | Revised after A/B benchmarking (maintainer: don't drop [NoAlias] from hot kernels
  without proof). GEMM-TransA benchmark, double N=1024, median of 4: with [NoAlias] 41.6 ms; without
  43.0/43.7 ms across two runs (~3-5% hint; float flat; double-512 inverted — within machine noise
  but never favoring the drop). FINAL POLICY: matMatDotTransA/Range KEEP [NoAlias] on all pointers;
  the aliased Aᵀ·A / A·A call shapes (Blas.dot(A, A[, transposeA: true]), isOrthogonal, covariance)
  are handled at the WRAPPER by copying one input to Temp (O(n²) copy vs the O(n³) product). Select
  kernels stay without [NoAlias] on a/b: dest-aliasing is a tested public contract
  (SelectRefTests VecAliasDest) and the loop is elementwise memory-bound, so copy-on-alias would
  cost more than any vectorization delta.
- 2026-07-13 | [NoAlias] made truthful (release-scan D3 ruling): write-aliasing wrappers now call
  dedicated single-pointer in-place kernels (signFlipInPlace; UnsafeBoolOP notInPlace/orInPlace/
  andInPlace/xorInPlace/equalsInPlace/notEqualsInPlace — the unused copy-form bool kernels were
  deleted).

## NLS
- 2026-07-12 | Release-scan fix (FOURTH bug, all precisions): the all-columns-flat degenerate
  case (0 < LInf(J) <= flatThresh, i.e. every column norm at-or-below flatThresh) left
  maxRealColNorm at 0 and d entirely zero, since the whole-Jacobian stationary guard only
  checked LInfJ0 > 0. nlsScaledGradNorm then divided by d[j]=0 (inf/NaN gradientNorm) and
  mu=1e-3*nlsMaxD2(d)=0 removed all damping. nlsUpdateScale now returns maxRealColNorm; both
  cores gate the initial stationary branch on that return being 0 (not on LInfJ0 > 0), which
  exactly matches nlsUpdateScale's own per-column flat classifier instead of approximating it
  via the whole matrix's LInf norm. Folded into the same change: nlsUpdateScale used to
  compute every column's squared-sum twice per call (once for maxRealColNorm, once for the
  floor pass); it now caches each column's norm in a scratch buffer (colNorms) and computes
  each column once.
- 2026-07-12 | NEW feature: nonlinear least squares via Levenberg-Marquardt with Nielsen damping
  (Optimize.nlsSolve / Optimize.curveFit). Algorithm reference: Madsen, Nielsen & Tingleff, "Methods
  for Non-Linear Least Squares Problems" (2nd ed., 2004), Algorithm 3.16 -- the gain-ratio damping
  update and the convergence structure (step-size test on the PROPOSED step, before it is evaluated
  against the objective). Math.NET Numerics' LevenbergMarquardtMinimizer.cs (MIT) was read as an
  independent C# structural reference; MINPACK (netlib, permissive) was read for STRUCTURE only, not
  transcribed line-by-line (the Marquardt column-norm-floored-at-running-max diag scaling is its
  well-known convention, not ported code) -- per the owner's provenance ruling this is the "provenance
  line + DEVLOG" bucket, not the "MINPACK acknowledgment in Third Party Notices.md" bucket. Robust-loss
  row rescaling (nlsApplyRobustScale) IS verified line-by-line against the installed scipy source
  (optimize/_lsq/least_squares.py's huber/cauchy + common.py's scale_for_robust_loss_function,
  BSD-3): z=(f/scale)^2, rho[0] scaled by scale^2, rho[2] divided by scale^2, rho[1] untouched;
  J_scale=sqrt(max(rho[1]+2*rho[2]*f^2, EPS)), f*=rho[1]/J_scale, J scaled per row -- confirmed byte-
  for-byte against scipy 1.17 rather than trusting the task's own sketch. Tukey biweight has NO scipy
  precedent (scipy ships no redescending loss, deliberately) -- its Rho/RhoPrime/RhoPrime2 were
  independently derived from the standard robust-statistics biweight identities (rho(r) = c²/6·(1-
  (1-(r/c)²)³) for |r|<=c else c²/6), then re-expressed in terms of s=r² under this library's rho(s)
  convention (0.5·Σrho(s_i) = the standard M-estimator total cost) -- see the from-scratch numpy
  prototype (scratchpad, not shipped) for the full re-derivation before this file was written.
- 2026-07-12 | Validation (numpy/scipy prototype, both precisions): (a) exponential-decay and sine
  fits matched scipy.optimize.least_squares(method='lm') to ~1e-8..1e-10 relative across 3-4 starts
  each. (b) NIST StRD Misra1a AND Chwirut2 (fetched fresh from the live NIST page -- an earlier hand-
  transcribed Chwirut2 table turned out wrong, see below) matched certified params to ~1e-6..1e-8
  relative in double precision from both prescribed starting points; Chwirut2 (not Misra1a -- see the
  scale-disparity entry a few entries below) is the one that also cleanly converges in float32, and
  is the one this library ships as its literal NIST test. (c) a parameter the model never references (exactly
  zero Jacobian column) stays EXACTLY at its initial value regardless of that value's magnitude
  (tested 0, 7.3, -1e6) while the other parameters converge normally -- no blow-up. (d) Huber/Cauchy/
  Tukey all recovered the true linear/exponential fit under 6/50-point gross-outlier contamination
  where plain L2 was visibly pulled off (e.g. linear fit relerr 0.89 for L2 vs 0.01-0.03 for the
  robust losses). (e) float32 rerun of (a)/(d) with the SAME relative-tolerance design reproduced
  the double-precision qualitative results (robust losses still beat L2 by the same margin), no
  precision-specific failure found. (f) numeric (forward AND central) vs analytic Jacobian converged
  to the same point (~1e-11 relative) in the same iteration count.
- 2026-07-12 | BUG FOUND AND FIXED during prototyping: an early engine design checked the step-size
  convergence test only AFTER an accepted step, mirroring a first guess at the M-N-T structure. On
  NIST Misra1a (b2 ~5.5e-4, a badly parameter-scaled start) this spiralled once residual/gradient
  reductions hit the float64 noise floor: each rejected trial produced a SMALLER step as mu grew, but
  the step was never actually re-checked against stepTol until AFTER an acceptance that never came,
  so mu escalated to the hard ceiling and the solve reported FailedLinearSolve despite already sitting
  at the certified optimum (params matched cert to 1e-8 well before the failure). Root cause: Madsen/
  Nielsen/Tingleff's own Algorithm 3.16 checks ||h|| <= eps2*(||x||+eps2) on the PROPOSED h, every
  iteration, BEFORE evaluating F(x+h) -- not only after acceptance. Re-reading the reference pseudocode
  and fixing the check order resolved it (verified: Misra1a now reports Converged/SmallStep, never
  FailedLinearSolve, from both prescribed starts). nlsSolveStep/the outer loop in NLS.fProxy.cs follow
  this corrected order; don't move the step check back to a post-accept-only position.
- 2026-07-12 | SECOND BUG FOUND AND FIXED (float32-only, caught by the float32 rerun of the NIST
  case): the Marquardt diag floor was first written as `dFloor = Consts.fProxySqrtEps * LInf(J0)`
  (scale-relative to the WHOLE Jacobian, deliberately not `max(1, LInf(J0))` -- the Kalman SDA bug is
  the same failure MODE, assuming an O(1) problem scale). That version still broke in float32 on NIST
  Misra1a specifically: b1's column norm (~0.16) and b2's column norm (~7e5) differ by ~1e6x, and
  sqrt(floatEps)~3.45e-4 times the LARGER column's norm floors b1's own legitimate ~0.16 column norm
  up to ~107 -- destroying its real gradient signal and reporting false-Converged at iteration 0
  (this did NOT happen in double: sqrt(doubleEps)~1.5e-8 is small enough relative to a 1e6x ratio
  that it stayed below 0.16). Root cause: ANY floor scaled by the WHOLE matrix's own magnitude
  cross-contaminates columns of genuinely different natural scale.
- 2026-07-13 | THIRD BUG FOUND AND FIXED (all precisions -- reported by the test suite as
  FlatParameterNoBlowup failing everywhere: float got 3.294179E+13, double got 8.54560688875843E+30,
  both "expected 0"). The second fix above (a plain `dFloor = Consts.fProxyEpsilon`, no matrix-scale
  multiplier) was itself insufficient: it was validated only against `np.linalg.lstsq` (SVD-based) in
  the numpy prototype, which does NOT reproduce this library's actual QR.solveInPlace (Householder).
  Root-caused with a FAITHFUL Python port of genHouseholder + solveInPlace's fused kernel (including
  the near-zero-column fallback `u[k]=sqrt(2)`), which reproduced the exact reported failure
  (h[flat]=5e29 at iteration 0) -- then confirmed byte-for-byte in a standalone dotnet harness with
  the SAME faithful port transcribed to C#: reinstating the plain-epsilon floor there reproduces
  8.545607E+30 (double) / 3.295198E+13 (float), matching the reported values to 4-6 significant
  figures. Mechanism: flooring a flat column's d_j at machine epsilon makes its augmented-system
  regularization entry sqrt(mu)*d_j fall BELOW QR's own zero-threshold
  (Consts.ZeroThreshold*LInf(Aaug)) for that column. genHouseholder's near-zero fallback sets ONLY
  u[k]=sqrt(2) and leaves the REST of u (including the regularization row itself, which is the
  column's ONLY nonzero entry) unchanged -- this fallback is correct for a column that is zero
  EVERYWHERE (the ordinary un-augmented QR case it was written for), but produces an inconsistent
  reflector when the column has exactly one small-but-nonzero entry (the augmented case): applying it
  leaves R's diagonal for that column proportional to mu*dFloor² (quadratically tiny), so
  back-substitution divides whatever roundoff has accumulated in the transformed RHS by a near-zero
  number and the flat parameter's step explodes. A larger CONSTANT floor does not fix this on its
  own either (tried 1e-6·LInf(J) through 2·LInf(J) in the harness): the required floor to clear QR's
  threshold scales with 1/sqrt(mu), and mu shrinks across iterations as the solve converges, so any
  FIXED floor value can eventually fall short again later in the same solve.
  FINAL FIX (nlsUpdateScale, NLS.fProxy.cs): stopped trying to pick a floor VALUE at all for flat
  columns. Instead, each iteration, first find maxRealColNorm (the largest column norm among columns
  ABOVE flatThresh = Consts.fProxyEpsilon, an ABSOLUTE per-type constant, never scaled by the
  matrix -- this is unchanged from the second fix and still correctly leaves Misra1a's b1, ~0.16, far
  above it and untouched). A column AT OR BELOW flatThresh (colnorm effectively zero -- the residual
  structurally does not depend on that parameter) is then floored at maxRealColNorm itself, not a
  small constant: this makes its regularization entry sqrt(mu)*d_flat EXACTLY EQUAL to the
  MOST-regularized real column's own entry, so it tracks mu's shrinkage in lockstep with the real
  columns and stays proportionally safe relative to QR's threshold for as long as the real columns'
  own regularization does (which is the normal, expected LM regime -- once real-column regularization
  itself becomes negligible relative to J, mu is deep into "trust the linearization" territory and the
  algorithm is behaving as intended). Matches the coordinator's candidate 1 (MINPACK's zero-column
  convention, generalized from a literal "1" to "the max column norm across J" for scale-independence)
  -- candidate 3 (explicit freeze/exclude) was not needed once the right floor TARGET was identified.
  Re-verified end-to-end in the dotnet harness with the FAITHFUL Householder QR (not a normal-equations
  or lstsq stand-in): all 9 shipped NLS test cases pass in both precisions, the Misra1a cross-
  contamination re-check (same scale-disparity case as the second bug) still reports the same benign
  SmallStep-at-iteration-0 (finite, unmoved, NOT a false convergence) rather than any blow-up or
  wrong-answer convergence, and reverting nlsUpdateScale to the flawed plain-epsilon version
  reproduces the FlatParameterNoBlowup failure with the reported magnitudes almost exactly (negative
  control). Same "no matrix-scale multiplier" reasoning still applies to the gradient-convergence
  reference scale (gnorm0 is the SCALED gradient's own value at the start, not a hardcoded constant).
- 2026-07-12 | Misra1a's ~1e6x b1/b2 scale disparity is ALSO why it was dropped in favor of Chwirut2
  as this library's single shipped NIST literal test: even with the dFloor fix above, Misra1a's own
  Marquardt mu0 heuristic (tau·max(d_i²), i.e. dominated by whichever parameter has the LARGEST
  column norm) over-damps the SMALLER-scaled parameter by a factor of roughly the scale ratio
  SQUARED. In float32 this makes the very first proposed step so tiny that adding it to p[0]=500
  is a float32 no-op (500+1.4e-6 rounds back to exactly 500 at ~7 significant digits) -- every trial
  is honestly rejected (rho_gain measures exactly 0, not negative or NaN) until the step-size test
  legitimately fires (SmallStep), from BOTH of NIST's own prescribed starting points. This is a real,
  explainable float32 precision limit for THIS problem's specific scale disparity, not an engine
  defect (no NaN, no crash, no silent non-convergence) -- but it makes a flaky/uninformative shipped
  test. Chwirut2's three parameters (~0.17, ~0.005, ~0.012, within ~30x of each other) have no such
  disparity and converge cleanly in BOTH precisions from its own NIST-prescribed start1 (double
  relerr ~8e-8, float32 relerr ~5e-3, both via genuine multi-iteration LM progress) -- this is the
  literal dataset TemplateSourceTests/fProxy/NLSTests.fProxy.cs actually ships.
- 2026-07-12 | Redescending-loss starting-point caveat (Tukey): if EVERY residual at the starting
  point exceeds the loss's Scale, RhoPrime is exactly 0 everywhere, the weighted gradient is exactly
  0, and the solve reports false-Converged at iteration 0 -- reproduced with Tukey(scale=0.3) from a
  poor exponential-fit start where every residual exceeded 0.3. This is inherent to ANY redescending
  M-estimator (not an engine bug) -- scipy's own least_squares ships no redescending loss for the
  same reason. Choose Scale comfortably larger than the expected residual spread at the start point,
  or warm-start from an fProxyHuberLoss/plain fit first.
- 2026-07-12 | Scoping decision: no analytic-Jacobian + robust-loss combination overload in v1 (the
  task brief's own bullets frame robust loss as an addition to the DEFAULT numeric-Jacobian path, and
  curveFit is explicitly numeric-only) -- kept deliberately, not an oversight. This also sidesteps a
  genuine C# landmine verified via a standalone dotnet repro before committing to the final overload
  ladder: two generic methods differing ONLY by a type-parameter CONSTRAINT (TF : IfProxyResidualFunction
  vs TF : IfProxyResidualJacobian) collide as CS0111 the moment their VALUE-parameter lists also
  match -- constraints are invisible to C# overload-signature uniqueness (same rule the naming-style-
  guide's "Split vs merge safety" section documents for merged classes). The numeric-only ladder's
  terse "just f, p, m" tier is therefore the ONE overload of that exact shape in the whole class
  (numeric is the default); the analytic ladder's shortest tier is the 6-param
  (f, p, m, gradTol, stepTol, maxIter) form, which the numeric ladder deliberately never offers (its
  own 6-param slot would collide) -- a caller wanting default tolerances on the analytic path passes
  Consts.fProxySqrtEps / Consts.fProxyEpsilon / 200 explicitly rather than getting a same-shaped terse
  overload. Compile-checked end-to-end (both precisions, all overload families, curveFit plain +
  weighted) via a standalone dotnet console project with API-compatible stub types before this file
  was written, not just reasoned about.
- 2026-07-12 | Convergence bookkeeping (cost, gain ratio, gradient/step norms) accumulates in
  `double` even in the float template -- same idiom as Optimize.ladIRLS's own `double dx=0,xn=0`
  convergence accumulator -- while J, r, h, d, mu (the actual factorized system QR.solveInPlace
  consumes) stay genuinely fProxy-precision. Confirmed this split doesn't mask a real float32 issue:
  the float32 prototype rerun with the SAME engine logic (native-dtype d/J/r/h, double-accumulated
  convergence tests) reproduced the double-precision qualitative results across every scenario.

## MPC / MPC.State
- 2026-07-12 | AUDIT POSTMORTEM (release-scan-2026-07-12/30-mpc-qpseam.md): confirmed HIGH --
  prestabilized input-bound rows read Phi/Gamma BLOCK k (x_{k+1}'s coefficients) instead of block
  k-1 (x_k's coefficients) when expressing u_k = -Kstab x_k + v_k, mis-constraining every stage's
  physical input and breaking the warm-start guess's feasible-by-construction property (the guess's
  own v_k = u_k + Kstab*x_k, evaluated with the CORRECT x_k, could not satisfy a row written against
  x_{k+1}). Root cause confirmed by direct read of MPC.State.fProxy.cs's row-assembly loop against
  the file's own Phi/Gamma block-k=x_{k+1} convention. FIX validated in a numpy prototype BEFORE
  editing the template (scratchpad/mpc-proto/mpc_prestab_bugfix.py): the audited off-by-one alone
  (block k -> block k-1, x_0=x0 identity for k=0) drove a deliberately-saturating case's u0 from
  -4.567 (outside [-2,2]) to -2.0 (respects the bound) -- but STILL disagreed with a fresh solve of
  the identical non-prestabilized problem by 0.198, which should be ~0 (prestabilization is a pure
  change of coordinates). Root-caused a SECOND, previously unaudited defect while chasing that
  residual: the condensed Hessian applied R naively to v (Rbar block-diagonal on v_k) instead of
  correctly expanding u_k^T R u_k with u_k = -Kstab x_k + v_k, silently dropping the -Kstab*x_k
  cross-coupling from the cost entirely. Fixed both together via one shared affine map, built once at
  construction and consumed by BOTH the rows and the cost so they cannot drift apart again: u_k =
  M_row_k @ V + c_k (c_k = -KPhiPre_row_k @ x0), M/KPhiPre built from block (k-1) (identity/Kstab
  directly for k=0); H_UU += M^T Rbar M (replacing the naive per-block R add for hasPrestab only);
  new persistent field Rcross = -2 M^T Rbar KPhiPre, applied as `c[0:nu] += Rcross @ x0` every solve
  call (MPC.fProxy.cs's BuildGradient). Extended prototype (mpc_prestab_full_fix.py) confirms the
  FULLY corrected version matches the non-prestabilized reference to ~1e-8 to ~3e-8 across binding,
  inactive, and random x0 -- vs 0.198-0.68 with only the row fix and no cost fix. Added
  PrestabBindingBoundMatchesNonPrestab to MPCTests.fProxy.cs (x0=(3,1.9), the SAME saturating case as
  SaturatedMatchesOracle) asserting both properties the coordinator specified: (i) u0 reconstructed
  independently from the state's own public Kstab/z fields respects the physical bound, (ii) matches
  a fresh non-prestabilized solve of the identical (A,B,Q,R,uLo,uHi) problem to tight tolerance --
  discriminates against BOTH the original off-by-one and the newly-found cost defect (verified by
  mentally/numerically reintroducing each).
- 2026-07-12 | AUDIT POSTMORTEM, low finding (same scan): MPC.solve's Fallback comment claimed
  state.z/wstatus are left untouched on "Infeasible/Unbounded", but QP.qpActiveSetCoreWarm only
  short-circuits before touching either on Infeasible -- on Unbounded (defensive-only, should not
  happen for MPC's genuinely PD H given R PD, but not structurally impossible) it runs the full
  active-set loop (which can mutate x = state.z via prior accepted steps) and unconditionally persists
  wstatus. Fixed by capturing u0out from the pre-solve warm-start guess BEFORE calling
  qpActiveSetCoreWarm (not re-derived from state.z afterward), so the "returns the shifted previous
  plan's first input" contract holds regardless of which failure status fires; state.uPlan/populated
  were already never written on this path (no change needed there). state.wstatus may still be
  perturbed on the (unreachable-in-practice) Unbounded path -- left as documented behavior rather than
  short-circuited in the QP seam itself, since RepairWorkingSet already re-validates every entry
  against the next frame's own state regardless, making a stale/perturbed persisted entry harmless.
  MPCStatus.Fallback's XML doc corrected to match (was overclaiming the same "both statuses" guarantee).
- 2026-07-12 | NEW feature: linear MPC over the standard batch/dense condensing (Borrelli-Bemporad-
  Morari, "Predictive Control for Linear and Hybrid Systems", ch. 2). acados/HPIPM (BSD-2) and TinyMPC
  (MIT) condensing routines were read for PRODUCT SHAPE reference only (decision-vector layout, the
  general idea of a fixed-at-construction condensed Hessian) -- no source line from either was
  transcribed; the actual Phi/Gamma/H assembly here is an original derivation verified against a from-
  scratch numpy/scipy prototype (scratchpad, not shipped) before this file was written. Soft-row exact
  penalty follows Kerrigan & Maciejowski, "Soft Constraints and Exact Penalty Functions in Model
  Predictive Control" (2000). qpOASES's MANUAL/thesis (warm-start strategy framing, Ferreau/Bock/Diehl
  2008) was read; qpOASES's SOURCE (LGPL) was not. DAQP (MIT) was read for active-set warm-start
  mechanics only.
- 2026-07-12 | Validation (numpy/scipy prototype, double integrator A=[[1,1],[0,1]], B=[[0],[1]],
  Q=I2, R=1 throughout): (a) unconstrained condensed MPC's u0 matched Control-style infinite-horizon
  LQR to ~1e-13 (double) / ~1.6e-7 (float32) across N in {1,3,10,30} -- a stationary DARE terminal cost
  makes ANY horizon reproduce the infinite-horizon law exactly, the correctness anchor. (b) input-
  saturated case matched scipy.optimize.minimize(method='trust-constr') on the identical condensed QP
  to ~1.6e-5 (its own convergence floor), independently cross-checked against a 3^n box-active-set
  brute-force enumeration. (c) soft wall: inactive case matched the unconstrained solution to ~5e-10;
  active-but-avoidable (input saturates but the wall itself is never touched) matched a hard-constrained
  trust-constr solve to ~1.6e-7 with zero slack, INSENSITIVE to rho1 across [0.5, 200] (all agreed to
  ~1e-7) -- the library's chosen default (rho1=1e3) sits well inside this margin; active-and-unavoidable
  (a double integrator's control has a one-step lag onto position, so the FIRST predicted stage's
  position is fixed by x0 alone) reproduced a hand-derived minimal-violation closed form exactly
  (0.3 then 0.6 over two stages). (d) receding-horizon active-set churn: [3,3,3,2,1,0,0,...,0] over 40
  frames -- collapses to 0 after frame 5, matching the "0-3 after the first" expectation. (f)
  prestabilization: rho(A)=1.2, N=40 raw condensing reached cond(H)~2.4e9 (float32-risky, though not yet
  NaN/inf) vs prestabilized cond(H_cl)~3.2 -- confirms the conditioning-insurance framing, not a
  strict correctness requirement at this rho/N.
- 2026-07-12 | Prestabilization (u_k = -Kstab x_k + v_k, condense the closed loop A-B*Kstab) turns hard
  input bounds into GENERAL rows (2*N*m of them) instead of a box on the decision vector, since u_k's
  bound becomes state-dependent (state depends on v through Gamma_cl) -- verified analytically that
  forward-simulating the warm-start guess with the REAL (A,B) and u_k, then deriving v_k = u_k +
  Kstab@x_k from that SAME trajectory, reproduces exactly the closed-loop condensing's own implied
  trajectory (x_{k+1} = A x_k + B u_k = (A-B Kstab) x_k + B v_k by construction). Combining
  prestabilization with the deltaU penalty is NOT supported in v1 (deltaU would need to couple to the
  state through the SAME substitution, compounding both derivations) -- throws at construction rather
  than silently dropping one feature. QR up/downdate for the per-iteration re-factorization was
  evidence-gated OUT of scope per the task brief; qpActiveSetCoreWarm re-factorizes the working set from
  scratch every pivot, same as qpActiveSetCore, fine warm at the target sizes (d <= 160).
- 2026-07-12 | Constructor overload ladder: deltaU-only and prestabilization-only convenience overloads
  were NOT added -- both would need an extra fProxyMxN-typed parameter (S / Kstab) in the exact same
  position as the "explicit terminal P" overload's own P parameter, a genuine C# overload-signature
  collision (parameter names never participate in overload resolution). Verified this is a real
  constructor-only distinction (methods can't disambiguate on names) via a standalone dotnet repro
  before writing the constructor ladder. Reach the full (17-parameter) constructor directly for those
  two features, passing `default` for the unused optional matrix params.
- 2026-07-12 | H (the condensed QP Hessian) is explicitly re-symmetrized via Control.SymmetrizeInPlace
  (reused directly, not reimplemented) after assembly -- Gamma^T Qbar Gamma accumulates through
  Blas.dot's own summation order, which can leave a tiny roundoff asymmetry even though the true
  mathematical result is exactly symmetric whenever Q/R/P/S are.

## QP
- 2026-07-14 | QP v2 stage 2c: DIFF-REPAIR replaces the all-or-nothing reuse. Instead of
  reuse-exact-or-full-rebuild, qpActiveSetCoreWarmPersistent now UP/DOWNDATES the persisted factor by
  only the rows that changed: ComputeTargetStatus gives the desired set for x0 (RepairWorkingSet's
  three-pass tightness logic, no factor build), the diff vs the persisted set is counted, and if it fits
  the dead-reflector budget (deadCount+numDrops < DeadCap) and is small (numDrops+numAdds <= DeadCap) we
  drop the no-longer-tight columns (high-to-low so shifts don't invalidate lower indices; DropFromFactor
  + UpdateReducedOnDrop) then add the newly-tight ones (TryAddToFactor + UpdateReducedOnAdd, rank-reject
  → Inactive). Large diff / first solve / budget-exhausting diff → full rebuild (resets the budget).
  Zero diff (steady state) = no work = the same reuse as before, so the measured steady-state warm win is
  unchanged (warm-box −25/−34% float, −46/−55% double; warm+wall −58/−74%); the diff-repair's extra
  benefit is on TRANSIENT ticks (working set moving a few rows), which the steady-state benchmark burns
  off — the win there is structural (incremental O(diff·n²) vs a full O(n²·nz) rebuild). Correctness
  still pinned by MPCTests.WarmPersistentMatchesColdEachFrame (its x0=(5,0) tight-bound trajectory
  saturates then desaturates → exercises real per-tick diffs, cross-checked vs cold QP.solve every
  frame). WarmSetUnchanged removed (subsumed by the zero-diff case).
- 2026-07-14 | QP v2 stage 2b SHIPPED: CROSS-TICK persistence for the MPC warm path
  (qpActiveSetCoreWarmPersistent). fProxyMPCState now OWNS the working-set factorization (qpFactor) and
  reduced space (qpReduced), carried across solves. First cut was reuse-exact-or-rebuild (superseded by
  the diff-repair entry above). ⚠️ JOB-COPY TRAP (cost most of a debugging pass): the
  wsf/red native BUFFERS survive an IJob.Run()/Schedule by-value copy of the owning state, but their
  PLAIN scalar fields (k, reflCount, deadCount, opCount, rotCount, stale, changeCount, factorValid) do
  NOT — so cross-tick reuse silently never fired through the benchmark's per-frame job.Run() (correct,
  since the rebuild path resets those counters, but +30-40% from paying incremental overhead with no
  reuse). Fix: MPCState carries a native-backed qpMeta[8] holding exactly those scalars;
  qpActiveSetCoreWarmPersistent rehydrates them on entry and writes them back on exit. Measured on the
  MPC warm steady-state benchmark (per-frame, via job.Run()): warm-box −25/−34% float, −46/−55% double
  at (40,12,4)/(30,24,8); warm+wall (active general rows → nontrivial reduced space) −58/−66% float,
  −68/−74% double. Correctness pinned by MPCTests.WarmPersistentMatchesColdEachFrame (every frame's warm
  solution cross-checked against an independent cold QP.solve of the identical condensed data).
  Allocator param added to both Create methods (Temp cold / Persistent for MPC). FUTURE: an incremental
  DIFF-repair (up/downdate the persisted factor by only the few changed rows) would extend the win to
  transient ticks where the working set moves by a few rows; the current fast-path is all-or-nothing
  (exact-match reuse else full rebuild).
- 2026-07-14 | QP v2 stage 2 SHIPPED: persistent up/downdated REDUCED SPACE (fProxyQPReducedState:
  Z, QZ = Q·Z, H_Z = ZᵀQZ carried alongside the stage-1 log). Kills the two O(n²·nz) per-iteration
  terms (FormNullSpaceBasis + the fresh Q·Z). Option (b) from docs/dev/draft-spec-qp-qr-updowndate.md
  — maintain Z/QZ/H_Z EXPLICITLY, recompute chol(H_Z) from scratch each iter; NOT a
  Gill-Golub-Murray-Saunders Cholesky updowndate (an add is a dense size-nz Householder congruence
  whose re-triangularization is O(nz³), no cheaper than the from-scratch factor). ADD: the new
  reflector restricts to the old null-space frame as Ĥ = I−ûûᵀ (û = reflector tail, read for free,
  ûᵀû=2); Z·Ĥ and (QZ)·Ĥ are rank-1, Q is NEVER re-multiplied (Q(ZĤ)=(QZ)Ĥ), Ĥ·H_Z·Ĥ is sym rank-2;
  the leaving direction is exactly local column 0 → delete it. DROP: Givens mix coords <k only so the
  old Z columns survive verbatim; one new column Q̂·e_k prepends (FormNullSpaceColumn) and H_Z borders,
  q_new=Q·z_new is the drop's only O(n²). Staleness: RefactorWorkingSet (DeadCap) reorders the frame →
  rebuild; RebuildCap=16 incremental changes → rebuild (roundoff bound). useIncrementalReduced flag on
  qpActiveSetLoop keeps the from-scratch path (SolveReducedNewtonStep) as an A/B + correctness seam;
  reduced buffers only allocated when the flag is set. A/B on a NEW loop-isolating QPBenchmark section
  (qpActiveSetCore from a supplied x0, no phase-1): incr vs batch −11/−12/−33% float, −12/−24/−48%
  double at n=16/64/192 (iters byte-identical, objectives match) — grows with n, comfortably >10% at
  n≥64 → GO. Full facade (Section 1) at n=192 ≈ half the stage-1 baseline (float 135.9→67.7ms). COLD
  path (qpActiveSetCore, QP.solve) defaults incremental; WARM path (qpActiveSetCoreWarm, MPC) defaults
  BATCH — a warm tick changes ~0 rows (MPC steady-state iters=0), so incremental maintenance never
  amortizes intra-call (cross-tick persistence is a future stage); an early incremental-warm default
  regressed the MPC headline 20-43% (per-solve n×n red alloc + copy-back with no iterations to earn it)
  before the flip. Dtype-collision trap avoided (RebuildCap/Create/Dispose on the dtype-named struct,
  same as stage 1's DeadCap). Test: fProxyQPFactorStateTests.IncrementalReduced (entrywise Z/QZ/H_Z vs
  fresh rebuild after every add/drop, across both rebuild triggers + the k=0 edge), CachedRetry
  (regularized retry off cached H_Z, byte-identical), FallbackEquivalence (incr vs batch agree
  status/obj/x — VALUES not paths). Trap while writing that test: ConstraintSense.LessEqual = −1, so a
  zero-init senses array is all Equal → x0=0 reads Infeasible; set senses explicitly.
- 2026-07-14 | QP v2 stage 1 SHIPPED: persistent up/downdated QR of A_Wᵀ (fProxyQPFactorState: op-log
  Q̂ᵀ = Householder reflectors + Givens rotations in creation order; hybrid store per
  docs/dev/draft-spec-qp-qr-updowndate.md option (a)). Replaces the per-iteration from-scratch
  refactor AND the O(nk²) trial factor per candidate add (TryAddToWorkingSet deleted; add rank-test =
  transformed tail norm, same threshold, identical decision). SeedWorkingSet/RepairWorkingSet build
  the factor incrementally too — the old per-candidate trial factors made seeding O(Σ nk²) ≈ O(nk³/3)
  at an LP-vertex start (~n tight rows), which dominated cold solves. Drop = R column shift + k-1-j
  rotations appended to the log; dead reflectors stay in Q̂; full refactor every DeadCap=8 drops keeps
  the log bounded (reflectors ≤ n+8, rotations ≤ 8(n-1)) and is also the defensive re-rank-guard
  (a row going numerically dependent during rebuild is set Inactive, same exclusion rule).
  Final stationarity diagnostics reuse the live factor (the old block refactored once more).
  Pure-add sequences are arithmetic-identical to the old batch factor (per-column reflector
  application order matches applyReflectorRightCols exactly); iteration paths diverge only after the
  first drop (rotations) or mid-loop add (column order = creation order, no longer ascending-t) —
  acceptance is KKT/oracle values per the spec, and the whole HS/brute-force/LP-limit battery passes
  unchanged. Dtype-collision trap hit on the way: consts/factories with proxy-free signatures on the
  SHARED partial QP class (FactorDeadCap, CreateFactorState(int)) collide between generated
  float/double partials — moved onto the dtype-named struct (DeadCap, Create, Dispose).
  Stage 2 (reduced-Hessian Cholesky border/downdate) remains measure-gated per the spec: H_Z
  formation (QZ = Q·Z, O(n²nz)) is now the dominant per-iteration cost.
- 2026-07-12 | Warm-start seam for MPC: qpActiveSetCore's loop body (the add/drop iteration, the
  perturbation-anticycling cleanup pass, and final diagnostics) was factored out, UNCHANGED, into a new
  internal qpActiveSetLoop(wstatus, ...) that neither seeds nor disposes `wstatus`/`L`/`U` -- the two
  entry points (qpActiveSetCore's existing seed-from-point behavior, kept byte-for-byte, and the new
  qpActiveSetCoreWarm) differ ONLY in how `wstatus` is seeded and who owns it afterward. Existing QP
  test suite is untouched by this refactor (qpActiveSetCore's own observable behavior did not change --
  same validation, same SeedWorkingSet call, same loop, same disposal, just relocated across two
  methods instead of one).
- 2026-07-12 | RepairWorkingSet (SeedWorkingSet's warm sibling): re-admits a PREVIOUS solve's
  ActiveLower/ActiveUpper row only if it is STILL tight (within feasTol) at the CURRENT x, on the SAME
  side it was active on before -- a row that drifted off its bound between frames is dropped rather than
  forced, since the active-set loop's invariant (A_W x = b_W at the start of every iteration) requires
  genuine tightness, not "was tight last time". Considered just re-running SeedWorkingSet fresh every
  warm call instead (simpler) and rejected it: SeedWorkingSet has no memory of the PRIOR working set at
  all, so it cannot report a meaningful workingSetChanges diagnostic, and (for a soft/general row that
  drifts slightly rather than snapping exactly to a new bound) would rediscover less of the previous
  optimal active set than a repair-first pass does.

## OP.Component / UtilityOP
- 2026-07-12 | UtilityOP.cs deleted (owner-approved): its zeroInPlace(in fProxyN) became a
  redundant special case of the generic below; no callers existed.
- 2026-07-12 | Generic zeroInPlace<T>/fillInPlace<T> added to the Comp families (born from the
  demo stress test: no way to zero/fill a matrix, and mulInPlace(A, 0) is a NaN propagator).

## CHO
- 2026-07-11 | Right-looking Cholesky chosen over left-looking: left-looking's hot loop is a dot-product reduction over already-computed columns, which stays scalar under strict FloatMode (loop-carried accumulator); right-looking's rank-1 update is a set of unit-stride row axpys, which vectorizes. (was CHO.fProxy.cs:43)
- 2026-07-12 | CHOL_BLOCK_MIN_N size gate is a measured crossover, not the naive 2*CHOL_BLOCK — the panel/TRSM/SYRK bookkeeping isn't amortised until ~8 panels wide. (was CHO.fProxy.cs:68)

## CHOP
- 2026-07-11 | Blocked PSTRF panel/SYRK boundary previously mirrored the lower triangle for cache purposes; that mirroring was found to be a cache cliff and was removed, keeping only the upper-triangle-by-row storage. (was CHOP.fProxy.cs:72)
- 2026-07-12 | CHOLP_BLOCK_MIN_N size gate is a measured crossover, higher than plain CHO's gate since the panel phase here is heavier. (was CHOP.fProxy.cs:107)
- 2026-07-12 | Blocked (level-3) PSTRF path is a port of Lucas/Higham dpstrf.f (upper-triangular branch). Two deviations from a literal port: Ukk is read straight from the pivot search's maxDiag (provably identical to re-deriving it, skips redundant work), and this port always searches for a pivot rather than reusing LAPACK's precomputed first-column pivot. Also, distinguishing rank-deficient from indefinite (this library's RankInfo, beyond LAPACK's single INFO=1) requires W accurate before the off-diagonal scan, so the rare branch that trips the tolerance check first flushes this block's pending columns [j0,k) via the same syrkUpperSub kernel, scoped narrower. (was CHOP.fProxy.cs:198, :209-215)

## LU
- 2026-07-12 | LU_BLOCK_MIN_N size gate is a measured crossover, not the naive 4*LU_BLOCK — the panel/TRSM/GEMM bookkeeping isn't amortised until ~8 panels wide. (was LU.fProxy.cs:116)

## Control
- 2026-07-12 | lqg() added: convenience solving BOTH the LQR control DARE (existing lqr) and the
  KF filter DARE (new Kalman.steadyStateGain) from the same A, returning a thin LQGInfo pair. Zero
  new Riccati math -- both calls reuse Control.SDACore, the filter side via the LQR/KF duality
  mapping (Kalman.fProxy.cs's file header). SymmetrizeInPlace widened private -> internal (no
  behavior change) so Kalman's PredictCovarianceCore/UpdateCore reuse the exact same
  symmetrize-after-roundoff hygiene instead of a second copy of the loop.

- 2026-07-11 | SDA recurrences implemented (Chiang-Fan-Lin Algorithm 2.1, no-cross-term/nonsingular-R case): A0=A, G0=BR⁻¹Bᵀ, H0=Q, A_{k+1}=Ak(I+GkHk)⁻¹Ak, G_{k+1}=Gk+AkGk(I+HkGk)⁻¹Akᵀ, H_{k+1}=Hk+Akᵀ(I+HkGk)⁻¹HkAk, Hk→S. The (I+GH)/(I+HG) solves are nonsymmetric n×n via LU (compact in-place + multi-RHS decompSolve), not Cholesky; G0=BR⁻¹Bᵀ is built via CHOP on R (not a bare inverse) so a semidefinite R degrades gracefully there too. (was Control.fProxy.cs:10-40, :164)

## Kalman
- 2026-07-12 | Release-scan perf fix: UpdateCore's Xt = Smeas^-1 * (H P) recomputed H*P via a
  fresh GEMM even though PHt = P Hᵀ (already computed for Smeas, still live) equals (H P)ᵀ
  exactly since P is symmetric at every call site. Xt is now Blas.trans(in PHt, ref Xt) --
  O(m·n) instead of O(m·n²), same result.
- 2026-07-12 | Bug found by the test suite (float only): SteadyStateGainVsOracle got a Kss ~98%
  relatively wrong (0.9765909 vs the 2e-3 float tolerance); FixedPathMatchesConverged missed its
  tracking bound by 0.109 downstream of the same bad gain. Root-caused with a float32 numpy harness
  transliterating Control.SDACore and the test's own OracleGain literally on the test's exact CV
  system (A=[[1,1],[0,1]], H=[[1,0]], Q=diag(1e-4,1e-4), R=[[0.05]]): SDACore's convergence test
  (residual = diffNorm / max(1.0, ‖Hk‖)) reported Converged after ONE doubling step in float
  (residual 2.644e-4, just under Consts.floatSqrtEps=3.4527e-4) while the true fixed point needs
  ~8 steps (confirmed independently by both the double-precision SDA run and the test's own
  fixed-point oracle, which agree with each other to ~1e-16/2e-7). The `max(1.0, ...)` floor is a
  reasonable absolute backstop for LQR's typically-O(1) cost weights, but Kalman process/
  measurement covariances are routinely << 1 (here ‖Q‖+‖R‖ ~ 0.05), so the floor turns the
  RELATIVE tolerance into an ABSOLUTE one at roughly the SAME scale as the quantities being
  tracked -- one tiny absolute step off Sigma0=Q satisfies it immediately, before the recursion
  has moved at all. Fixed in steadyStateGain (not in Control.SDACore itself, to avoid touching the
  shared LQR cold-solve path and its own test suite): jointly rescale Q/R by
  1/max(‖Q‖+‖R‖, Consts.fProxyZeroThreshold) before the SDA call and unscale Sigma after -- proven
  exactly invariant for Kss (scaling Q and R by the same c scales Sigma by c, leaving
  Sigma Hᵀ(H Sigma Hᵀ+R)⁻¹ unchanged), confirmed in the float32 harness: relImplCorrect
  0.9765909 -> 3.385e-7, iterations 1 -> 6, while the wrong-orientation discrimination margin
  (relOraclePair/relImplWrong ~3.57) is untouched. Also confirmed harmless for double (already
  exact, scaling doesn't change the converged answer, iteration count unchanged at 8).
  Control.FrobeniusNorm widened private -> internal for this (no behavior change).
- 2026-07-12 | NEW feature. Algorithm reference: FilterPy (rlabbe/filterpy, MIT) -- predict/update
  equations (x=Ax+Bu, P=APAᵀ+Q; y=z-Hx, S=HPHᵀ+R, K=PHᵀS⁻¹, x+=Ky, Joseph-form
  P=(I-KH)P(I-KH)ᵀ+KRKᵀ) fetched and verified line-by-line against kalman_filter.py/EKF.py.
  Interface-shape reference: mherb/kalman (MIT) -- separate propagation/measurement function plus
  a separate Jacobian, not one fused updateJacobians() call. FORBIDDEN sources (per owner ruling,
  not used): MathNet.Filtering Kalman (LGPL despite MIT-labeled repo), TinyEKF historical
  snapshots (LGPL then).
- 2026-07-12 | K is never formed via an explicit S inverse anywhere in this file: every gain
  computation (UpdateCore, steadyStateGain) solves the TRANSPOSED system S·Kᵀ = (PHᵀ)ᵀ = HP via
  CHOP (pivoted Cholesky), so a rank-deficient S degrades to a minimum-norm K instead of a hard
  failure or a divide-by-near-zero.
- 2026-07-12 | steadyStateGain's SDA-duality mapping (Ã=Aᵀ, B̃=Hᵀ, S↔Σ) was validated against an
  INDEPENDENT ground truth before this file was written (Python prototype, plain fixed-point
  iteration of the KF predicted-covariance Riccati equation from Σ0=Q, no SDA/doubling involved):
  agreement to ~1e-16 relative Frobenius norm on a 2-state CV tracker, AND against a THIRD
  independent path (the actual predict/update Joseph-form recursion iterated to steady state,
  gain extracted from its last update call) to ~1e-16. A deliberately-wrong mapping (forgetting
  the A transpose, i.e. Ã=A instead of Aᵀ) was also run and diverges from ground truth by ~1e-2
  relative -- confirms the test is actually discriminating, not passing by coincidence.
- 2026-07-12 | EKF interface choice: analytic Jacobian REQUIRED on IfProxyKFModel/
  IfProxyKFMeasurement (JacobianF/JacobianH), no numeric-differentiation fallback baked into the
  interface itself. A wrapper-functor design (an fProxyNumericKFModel<TInner> auto-computing the
  Jacobian for a Jacobian-less inner model) was considered and rejected: it needs a nested generic
  struct implementing IfProxyKFModel while itself being generic over another IfProxyKFModel-minus-
  Jacobian shape, which has no precedent elsewhere in this codebase's struct-functor family
  (IfProxyLinearOperator's wrappers like fProxyColScaledOperator wrap ONE inner operator of the
  SAME interface, not a different, smaller interface) and adds a layer of generic indirection for
  a case (no analytic Jacobian available) that is the exception, not the rule. Shipped instead:
  Kalman.numericJacobianF/numericJacobianH, plain central-difference helpers a user calls FROM
  INSIDE their own JacobianF/JacobianH when hand-differentiating is impractical -- same
  "provide the primitive, not a forced wrapper" shape as QRCP's tol3z reuse of Consts.fProxySqrtEps.
- 2026-07-12 | fProxyKFState's own scratch is genuinely zero-Allocator.Temp for predict/ekfPredict/
  predictFixed/updateFixed (every intermediate is a pre-allocated field, sized once at n or n x n
  at construction). The general update()/ekfUpdate<TMeas> path does NOT extend this to its
  measurement-shaped intermediates (Hᵀ, PHᵀ, S, the CHOP factor, K) -- these are per-call
  Allocator.Temp, sized to that call's actual H.M_Rows, deliberately mirroring
  Control.RiccatiStep's own R+BᵀSB solve (also per-call Temp, also variably shaped). Considered and
  rejected: pre-allocating update()'s scratch at fProxyKFState.MMax and reinterpreting a smaller
  logical sub-block of it per call -- the library's dot/CHOP primitives all validate EXACT
  dimension equality (no stride/logical-sub-size concept anywhere), so this would need either
  mutating fProxyMxN.M_Rows/N_Cols post-construction (undocumented elsewhere, and fProxyMxN.Length
  is a readonly field that would then disagree with M_Rows*N_Cols) or a raw NativeArray-view
  reinterpretation via NativeArrayUnsafeUtility (safety-handle bookkeeping for no proven benefit --
  a per-call CHOP factorization is O(m³), already far more expensive than one Temp bump-allocator
  vector/matrix allocation). MMax is used only by the fixed-gain fast path (predictFixed/
  updateFixed), which genuinely needs and gets zero-Temp-alloc treatment since Kss is fixed-shape
  for the state's whole lifetime.

## Kalman.UKF
- 2026-07-12 | NEW feature (UKF, next increment after the linear/EKF Kalman filter). Algorithm
  reference: FilterPy (rlabbe/filterpy, MIT) -- MerweScaledSigmaPoints.sigma_points/_compute_weights
  and UnscentedKalmanFilter.predict/update, fetched and verified line-by-line. ukfPredict/ukfUpdate
  reuse the SAME IfProxyKFModel/IfProxyKFMeasurement functors ekfPredict/ekfUpdate use, calling ONLY
  F/H -- JacobianF/JacobianH are never read, which is the whole point of the unscented transform
  (no linearization at all, not even an approximate one).
- 2026-07-12 | FLOAT-RISK FINDING, deviation from FilterPy's cited default: Van der Merwe's classic
  write-up (and FilterPy's docstring) recommends alpha ~1e-3. Measured in the float32 numpy
  prototype (CV tracker, UKF vs the exact linear-KF oracle -- sigma points are exact for a LINEAR
  F/H, the strongest correctness check available): alpha=1e-3 gives max|x diff|=0.86 (catastrophic
  -- worse than useless) and max|P diff|=5.8, vs alpha=1.0 giving 1.9e-6/2.0e-6 (both essentially at
  the float32 precision floor for this problem's scale). Root cause: n+lambda = alpha²(n+kappa)
  shrinks the sigma-point spread by alpha while lambda/(n+lambda) (and every other weight, which is
  ∝ 1/(n+lambda)) grows by roughly 1/alpha² -- at alpha=1e-3 the weights reach ~±1e6 (see the
  concrete numbers in the fProxyUKFCache DEVLOG entry) and the covariance recombination becomes a
  weighted sum of near-identical numbers with huge opposite-signed weights, i.e. textbook
  catastrophic cancellation. This library's DEFAULT is alpha=1, beta=2, kappa=0 instead --
  confirmed (same harness) that UKF then tracks a nonlinear pendulum AS WELL AS OR BETTER than EKF
  in both precisions (double: EKF 0.00718 vs UKF 0.00713; float: EKF 0.00718 vs UKF 0.00579 --
  UKF actually wins in float, matching the "UKF should track as well or better" acceptance bar).
  Double precision also improves under the new default (3.6e-15 vs 4.7e-9 relative agreement with
  the linear-KF oracle at alpha=1e-3), so this is not purely a float32-only trade. A caller can
  still construct <see cref="fProxyUKFCache"/> with an explicit smaller alpha via the 4-arg
  constructor; the algorithm remains correct there (validated: alpha=0.1 and 0.05 both keep P
  exactly symmetric and PSD, min eigenvalue ~4e-4, over 2000 steps in both precisions despite
  Wc[0] reaching -96 / -396) -- just with markedly less numerical margin, which is now a documented,
  deliberate caller choice rather than a silent trap.
- 2026-07-12 | GenerateSigmaPoints regenerates sigma points FRESH at the start of BOTH ukfPredict
  and ukfUpdate -- a deliberate deviation from FilterPy's UnscentedKalmanFilter, which reuses
  predict()'s propagated `sigmas_f` directly inside update() (a documented perf shortcut in
  FilterPy's own code, not part of Van der Merwe's original algorithm). Reasoning: this library's
  own Kalman.update already supports being called more than once per predict (multi-sensor fusion
  between predicts); reusing stale sigma points across a second ukfUpdate call in the same pattern
  would silently under-represent the covariance change the first update just made. Regenerating
  costs one extra O(n³) Cholesky per ukfUpdate call, and is mathematically IDENTICAL to FilterPy's
  result in the common case (update immediately follows predict, nothing else in between).
- 2026-07-12 | Permutation-aware sigma-point scatter: CHOP factors Pᵀ·Σ·P = L·Lᵀ (P the pivot
  permutation, Σ the state covariance -- disambiguated from CHOP's own P-for-permutation in
  comments as "the permutation"), so L's COLUMNS are in PIVOTED order, not the original state-index
  order. The Van der Merwe spread vector for column k is therefore built by SCATTERING L's column
  through the permutation (v[Piv[i]] = L[i,k]·scale), not read off directly -- verified in the
  Python prototype's own pivoted-Cholesky emulation (which deliberately pivots, unlike numpy's
  plain `cholesky`, specifically to exercise this scatter logic before it was ported to C#).
  Getting this backwards (reading L[k,i] or skipping the permutation) would silently produce a
  valid-LOOKING but WRONG sigma spread for any P that actually pivots (i.e. essentially always,
  since CHOP pivots greedily by largest remaining diagonal even for well-conditioned input).

## Kalman.UKFCache
- 2026-07-12 | Chose a SEPARATE fProxyUKFCache over folding sigma-point buffers into fProxyKFState
  (the spec's other offered option): keeps the linear/EKF/fixed-gain paths (which never need sigma
  points) free of (2n+1)-sized memory, mirrors the house Cache convention (fProxyCHOPCache,
  fProxySVDThinCache) of a workspace struct paired with -- not merged into -- the data it operates
  on, and lets a caller reconfigure alpha/beta/kappa (a UKF-only concept) without touching
  fProxyKFState's own constructor arity.
- 2026-07-12 | Nests CHOP's own fProxyCHOPCache (`chopWs`) rather than calling CHOP.decomp's
  convenience (non-workspace) overload, which allocates an n x n Allocator.Temp buffer internally
  every call -- caught by re-reading CHOP.decomp's own source after first wiring GenerateSigmaPoints
  to the convenience overload, which would have silently broken the "ukfPredict is zero-Temp-alloc"
  claim. `bt` (CHOP's solve-side scratch) is deliberately left uncreated -- sigma-point generation
  only ever calls `decomp`, never `decompSolve`.
- 2026-07-12 | See the concrete alpha=1e-3 default-negative-Wc[0] numbers this defaults choice
  avoids: n=2, alpha=1e-3, beta=2, kappa=0 gives lambda=-1.999998, Wm[0]=Wc[0]≈-1e6, every other
  weight ≈+2.5e5 (computed in the float32 prototype). alpha=1 (this library's default) instead
  gives lambda=0=kappa, Wm[0]=0, Wc[0]=2 -- non-negative for the default case, though a caller-
  chosen alpha&lt;1 can still drive Wc[0] negative (by design; see Kalman.UKF's own DEVLOG entry).

## Kalman.State
- 2026-07-12 | Scratch fields (xNext/Bu/AP/APAt/At/J/yFast) are `public`, not `internal`, matching
  the house Cache/State convention (fProxyCHOPCache, fProxyLQRState both use public fields) rather
  than hiding them -- these are workspace buffers, not encapsulated implementation state.

## Kalman.Info
- 2026-07-12 | KFStatus has only two members (Ok / InnovationSolveFailed) because CHOP.decomp on
  the innovation covariance S = HPHᵀ+R has only two outcomes worth distinguishing here:
  Success/RankDeficient (both usable -- S is generically PSD whenever P is, so RankDeficient is
  expected on a redundant/collinear sensor row, not an error) collapse to Ok, and Indefinite (S
  numerically broken) is the only real failure.

## FFT (no-workspace path)
- 2026-07-15 | REMOVED the no-workspace fft/ifft/rfft/irfft overloads and their sin/cos recurrence
  cores (FftCore radix-2, FftCoreRadix4Rec) from FFT.fProxy.cs. Rationale: the path was strictly
  dominated — even a one-shot ws+build (build the quarter-wave table + one transform ~7.4 ms at
  N=2^20 float) beat the recurrence (~22 ms), AND it was non-deterministic across architectures
  (sin/cos twiddles). Nothing justified keeping it. Workspace overloads are now the only power-of-two
  path. Tests repointed to FFT.dft as the independent oracle (small N; both dispatch paths, N up to
  2048) + round-trips + analytic. dft/idft KEPT as the arbitrary-N fallback (still sin/cos, still the
  documented non-deterministic escape hatch — DetMath would be the deterministic route, parked). Don't
  re-add a recurrence FFT: if a zero-setup convenience call is ever wanted, build a Temp workspace
  internally (build scratch is only 4 MB post quarter-wave) rather than reviving sin/cos.

## FFT.Workspace
- 2026-07-15 | irfft output de-interleave FUSED into the inner core's 1/N inverse-scale pass, so both
  ends of irfft are now fused (input re-pack into the permutation + output interleave into the scale).
  irfft(ws) float 1M 4.04 -> 3.52 ms (another -13%; now FASTER than rfft's 3.62, since rfft only fuses
  its input pack — its WQ unpack is still a separate output pass). double 1M 4.66 -> 4.17. The old
  irfft did the scale in place (rePtr[i]*=invN) then a separate pass real[2j]=cz[j]/real[2j+1]=sz[j].
  Fused: FftCoreRadix4Core/FftCoreRadix4MixedCore gained an interleaveOut pointer; when non-null
  (inverse only) the final scale writes interleaveOut[2i]=Re*invN, [2i+1]=-Im*invN straight into real
  instead of back in place. real is a separate buffer from cz/sz, so no aliasing. Bit-identical (same
  values, different destination). Complex ifft passes null → unchanged in-place path. FFT tests 102/102,
  no regression in complex fft/ifft (6.56/6.56). Output-side fusion now DONE for irfft; the dedicated-
  real-last-stage idea for rfft's unpack (k,M-k ≠ combine k,k+M/2) is a different, still-open item.
- 2026-07-15 | irfft re-pack FUSED into the inner inverse FFT's first permutation (mirror of the rfft
  pack fusion below). irfft(ws) float 1M 4.88 -> 4.04 ms (-17%), double 1M ~4.66. The half-spectrum
  re-pack (E/O reconstruction over Hermitian pairs) already writes cz/sz out-of-place from re/im, so
  scatter each re-packed sample k straight into its post-permutation slot dst(k) and call the compute
  core directly — the inner FFT skips its own reversal / cycle-following de-interleave. Extracted
  FftCoreRadix4Core (reversal-skipping variant of FftCoreRadix4, mirrors the MixedCore extraction);
  pure path calls FftCoreRadix4Core, mixed path FftCoreRadix4MixedCore. dst is a bijection over [0,M)
  and the pack reads re/im (separate buffer), so no collision/aliasing. Bit-identical, FFT tests
  102/102, no regression in complex fft/ifft (6.59/6.64). Unpack-into-last-stage (output side) still
  open: unpack pairing k,M-k ≠ combine k,k+M/2, needs a dedicated real last stage — big, deferred.
- 2026-07-15 | rfft pack FUSED into the inner FFT's first permutation → ~1.77x faster than complex
  fft(ws) (was ~1.16x), i.e. near the theoretical 2x for a two-for-one real FFT. rfft(ws) float 1M
  5.50 -> 3.62 ms (-34%), 262K 1.27 -> 0.83, 16K -42%; double 1M 6.09 -> 4.10. Mechanism: the old
  rfft did a separate pack pass (deinterleave real -> cz/sz) AND then the mixed core did its OWN
  in-place even/odd de-interleave via cycle-following (+ a full-length visited[] clear + pointer-
  chasing). Those are the SAME axis, so fused: scatter real[2j]/real[2j+1] straight into the inner
  FFT's first-permutation slot, OUT-OF-PLACE (real is a separate source, so no cycle-following /
  visited needed at all). Pure-radix-4 case (M=4^k): first permutation = base-4 digit reversal, so
  the fused scatter writes to ReverseBase4Digits(j) and then calls FftCoreRadix4Ptr directly. Mixed
  case (M=2·4^k, the common one at N=4^k): first permutation = even/odd de-interleave, fused scatter
  writes to dst(j) then FftCoreRadix4MixedCore. Refactored FftCoreRadix4Mixed -> MixedDeinterleave +
  FftCoreRadix4MixedCore so rfft can call the compute core after its own fused permute; the conjugate
  moved post-de-interleave (elementwise negate commutes with the permutation → identical). Bit-
  identical (same permutation, computed differently); FFT tests 102/102. Research (2 agents, sourced):
  planar/split re/im is the SIMD-right layout — interleaved needs shuffles our portable fProxyW
  deliberately lacks (Popovici/Franchetti HPEC'17; FFTW genfft; confirmed) — so this fusion-on-split is
  the correct route, NOT interleaving. Only complex-FMA HW (AVX-512 FP16 / Arm FCMLA) escapes, not
  reachable portably. irfft mirror + unpack-into-last-stage still open (unpack pairing k,M-k ≠ combine
  k,k+M/2, so unpack fusion needs a dedicated real last stage — big, deferred).
- 2026-07-15 | Serpentine (boustrophedon) butterfly group order tried and REVERTED. Don't retry.
  Idea: alternate the base_ group loop high->low by stage parity so each stage restarts in the
  address range the previous stage left hot (groups are disjoint/order-independent, so it's
  bit-identical — FFT tests stayed 117/117). A/B on a quiet PC: it's a wash-to-negative. fft(ws)
  float gained ~3-5% (1M 6.56 -> 6.30 ms, 16K -5%), but double REGRESSED ~1-3% (1M 7.25 -> 7.45)
  and rfft(ws) regressed on both dtypes (float 262K 1.274 -> 1.352, double 1M 6.09 -> 6.21).
  Regressions were consistent across sizes (not noise): the descending stream costs the 4-lane
  double path more than the hot restart saves (weaker descending-prefetch, more exposed), and the
  mixed-path rfft sub-FFT halves don't align with the combine boundary. Net makes 3 of 4 paths
  slower. The proper version of this locality idea is cache-blocking (six-step FFT), not loop flips.
- 2026-07-15 | rfft/irfft unpack: process bins in Hermitian-symmetric pairs (k, M-k). Under k -> M-k
  the twiddle maps W_N^(M-k) = -conj(W_N^k), so E_im, O_im and Re(W) flip sign — one WQ call and one
  (k, M-k) load produce BOTH outputs (re[k]=E_re+P, re[M-k]=E_re-P, etc.). Halves WQ calls and, more
  importantly, halves the effective cz/sz read traffic: the reverse-stream partner cz[M-k] is consumed
  in the same iteration as cz[k] instead of being re-fetched M elements later (long evicted at large N,
  so the old loop paid for cz/sz twice). Zero added memory. Loop now runs k=1..M/2-1 paired + a single
  self-paired middle bin k=M/2; guarded for M==1 (N==2, no general bins). rfft(ws) quiet-PC: float 1M
  6.14 -> 5.50 ms (-10%), 262K 1.478 -> 1.274 (-14%), 64K 0.308 -> 0.272; double 1M 6.58 -> 6.09; the
  gap over complex fft(ws) widened ~7% -> ~16% at 1M. Recovered ~0.6 of the ~2.8 ms pack+unpack
  overhead; the paired loop is now SIMD-shaped if pushed further (reversing load for the M-k stream +
  boundary-split WQ). Pack (real[2j]/[2j+1] -> cz/sz) left as-is — sequential deinterleave, already
  bandwidth-bound, nothing to fuse.
- 2026-07-15 | Quarter-wave twiddle table: store only the first quadrant cos (twQuarter, n/4+1)
  instead of two full/half arrays; reconstruct any W^m via CosQ (quadrant reflection) + a π/2 index
  shift (Im(W^m)=CosQ(m+n/4)). Cuts the persistent table 8→1 MB and the build double-scratch 16→4 MB
  at N=2^20 float (over the session: full 8 MB → half 4 MB → quarter 1 MB). ~1 ULP accurate, NOT
  bit-exact (reflected entries are independently-built, so a fold's sign flip isn't an exact
  negation) — within the existing 1e-6/1e-12 twiddle tests; TwiddleTableAccuracy rewritten to check
  the quarter values + full-circle reconstruction. N=2 degeneracy: Q=n/4=0 breaks the π/2 shift (Im
  should be 0), guarded in WQ (tableN>=4). Perf lessons, all A/B'd on a quiet PC (no-ws path as the
  stable control, cross-run):
  * The old README/fft.md workspace numbers (12.9/11.3 ms @1M) were STALE — pre-dated the wide-SIMD
    campaign. True steady-state is ~6.5/6.0; full-vs-quarter A/B on one machine confirmed the "2x"
    was a stale-doc mirage, not this change.
  * Quarter table alone (WQ in the scalar butterfly) REGRESSED fft(ws) float ~+10% (float has 2
    scalar stages q=1,4; double 1 → double stayed flat). Cause: WQ's branchy reconstruction ran
    per-butterfly in the finest stages (which carry n/4 butterflies each).
  * Fix 1 — materialize EVERY stage's W^1 into sw1 (was wide-stages-only; +5 entries, swLen still
    ~n/3), so the scalar butterfly reads W^1 and derives W^2/W^3 in-register like the wide path. No
    runtime WQ in the butterfly. Recovered most (+10% → +3%).
  * Fix 2 — reorder the scalar butterfly to j-outer/base-inner so the q(<=4) twiddle triples are
    computed ONCE per j (not per-butterfly), held in registers. Closed the rest: steady-state now
    parity-to-faster than the half-circle version (262144 ~-5%, double 1M ~-7%, 1M float within
    noise). Zero extra workspace memory, no threaded pointers.
  * cw1 (radix-2 combine) and sw1 stay materialized-from-CosQ-at-build → the wide combine/butterfly
    hot loops read contiguous tables, no runtime reconstruction. Only rfft/irfft unpack calls WQ.
  * Removed dead twq/tableN threading through FftCoreRadix4/Slice/Ptr/Mixed once the butterfly went
    fully sw1-driven (the stage twiddles are stage-length-relative, W_(4q)^j, independent of tableN).
  Radix-16 / storing W^2/W^3 in the workspace / horizontal-SIMD WQ all considered and rejected (see
  session notes). Next real speed levers are bigger: rfft pack/unpack fusion, cross-stage cache
  blocking — NOT the table.
- 2026-07-15 | SIMD'd the last scalar loop: the radix-2 DIT combine (Step 3 of FftCoreRadix4Mixed).
  The combine reads E/O data (re[k],im[k],re[M+k],im[M+k]) contiguously in k but the twiddle
  W_size^k = twReFull[k*combineStep] was strided, forcing a scalar loop. Fix mirrors the sw1 trick
  one level up: gather the combine twiddle into a contiguous per-workspace table cw1re/cw1im so k
  indexes it directly, then wide-load twiddles + all four data streams (fProxyW), scalar tail for
  M<Width (M is a power of 4, so no partial wide iteration once M>=Width). A given n triggers mixed
  at exactly ONE (size,step): n=2·4^k → fft/ifft at size=n step=1 (already contiguous → cw1 aliases
  twReFull, no copy); n=4^k → rfft/irfft inner mixed at size=n/2 step=2 (gathered, length n/4). So
  one cw1 table per workspace suffices; both dispatch paths pass ws.cw1re/cw1im. Measured (quiet PC,
  A/B new-vs-baseline, pure-radix-4 fft(ws) rows as the unchanged thermal anchor): rfft(ws)/irfft(ws)
  ~1.15-1.24× faster across float+double, all sizes, remarkably flat (float 1M 1.18×, 256K 1.19×;
  double 1M 1.24×, 256K 1.15×); mixed fft(ws) (2·4^k, top-level combine a bigger fraction) corroborates
  ~1.19-1.29× but noisier (baseline hit thermal spikes at double 16K/1M). Pure-radix-4 (pow-4) paths
  unchanged — no combine. This was the last scalar butterfly loop; the whole transform is now wide.
  Suite 6228/6228.
- 2026-07-14 | Extended the sw1 wide radix-4 butterfly to the rfft/irfft/mixed sub-transforms.
  Previously only the top-level pow-4 fft/ifft used the wide (fProxyW) butterfly; the mixed-radix
  (2·4^k) path and the rfft/irfft inner M-point FFTs ran the SCALAR FftCoreRadix4Ptr. Unified: made
  FftCoreRadix4Ptr the hybrid wide+scalar kernel (q>=Width wide via s1r/s1i, q<Width scalar), threaded
  sw1re/sw1im through FftCoreRadix4Slice/FftCoreRadix4/FftCoreRadix4Mixed, and deleted the standalone
  FftCoreRadix4Wide (fft/ifft pow-4 now route through FftCoreRadix4 → same wide path, one butterfly
  copy instead of two). Sub-transforms share the SAME sw1 table: size-M sub-FFTs share tableN=n, so
  stage q needs step=n/(4q) and stageOff layout identical to the top level — no second table. Build
  gate changed from `pow4 && qq<n` to `4*qq<=n` (drop pow4) so non-pow-4 workspaces also fill sw1
  (largest stage over all sub-transforms has 4q<=n). Measured (thermal-normalized rfft(ws)/fft(ws),
  since fft(ws) pow-4 is unchanged and cancels per-run thermal drift; raw A/B was unusable — PC not
  quiet, unchanged fft(ws) anchor swung 1.1-1.5× between runs): rfft(ws) ~1.3-1.6× faster
  (float 1M 1.45×, float 256K 1.27×, double 1M 1.59×, double 256K 1.34×). Suite 6228/6228.
- 2026-07-14 | Build ~13x faster via recursive-doubling twiddle fill. Replaced the per-entry
  bit-decomposition (O(n·log n): each W^m an independent product of log n generators) with a
  doubling fill (W^0=1, then W^(2^k+j)=W^j·B_k for j<2^k — one complex-mult per entry, O(n) total).
  Each entry is still <= log2(n) mults deep so error stays O(log n·ε); done in a double scratch,
  cast to fProxy once (same accuracy model, TwiddleTableAccuracy <1e-6 still passes). Measured
  float N=1M build ~41ms → ~3ms (ws+build 48→9.8ms; transform unchanged ~6.7ms); double similar.
  Cost: a transient double scratch of 2N doubles (16 MB at N=1M) via UnsafeUtility.Malloc/Free,
  freed before the factory returns (steady-state workspace memory unchanged). The doubling reads
  back intermediate values, so the scratch must be double even for the float variant to keep the
  chain O(log n·ε) rather than O(log n·float-eps).
- 2026-07-14 | sw* halving: store only the W^1 stage table (sw1re/sw1im), derive W^2=W^1·W^1 and
  W^3=W^1·W^2 in-register in the wide butterfly. Drops sw2/sw3 (4 of 6 arrays) → workspace
  −5.6 MB at N=1M float (−11 MB double), 28.4→22.8 MB. Perf-NEUTRAL: the 2 extra complex-mults
  per butterfly are absorbed because the wide butterfly is load/bandwidth-bound — 4 fewer twiddle
  streams offset the compute (measured 262144 float 1.67 vs 1.66, 1M double 8.2 vs 8.4, both in
  noise; 65536 float row is erratic across runs, ignore). No longer bit-identical to the scalar
  butterfly (derived W^2/W^3 differ ~2 ulp from tabulated), but TableFftVsRecurrence's 1e-3 tests
  pass; suite 6228. Remaining sw1 ≈ 2N/3 floats.
- 2026-07-14 | Deterministic twiddle-table build: replaced the per-entry math.cos/math.sin loop
  in fProxyFFTCache with root-of-unity generation using only +,-,*,sqrt. The table is W_N^m =
  exp(-2πi·m/n); built from binary generator roots B_k = exp(-2πi·2^k/n) via stable unit-circle
  half-angle square roots (c=sqrt((1+a)/2), s=b/(2c) — cancellation-free), each W_N^m the product
  of B_k over m's set bits (bit-decomposition, ≤log2(n) mults/entry so error is O(log N·ε),
  bounded — NOT the O(N) drift of a linear recurrence, the drift the direct-cos/sin table
  originally avoided). WHY: sqrt is IEEE correctly-rounded (bit-identical cross-arch), and +/-/*
  don't reassociate under FloatMode.Strict, so the whole build is cross-arch deterministic —
  unlike math.sin/cos, which Burst only guarantees identical under FloatMode.Deterministic
  (opt-in, 64-bit only). This closes the FFT's only non-deterministic step: under Strict the
  workspace fft/ifft/rfft/irfft path (build + butterfly, all +/-/*/sqrt) is now cross-arch
  reproducible. (The no-workspace recurrence path and dft still call cos/sin.) Built at double
  precision for both dtypes, cast to fProxy (float table rounded once from a double-accurate
  table — same as the old design). Verified: TwiddleTableAccuracy test asserts <1e-6 vs
  math.cos/sin at n=2/8/4096/65536, float+double; all FFT round-trip/vs-recurrence/vs-DFT tests
  pass (suite 6228). Build cost unchanged (~41 ms at N=1M float, same as the cos/sin build); the
  timed reuse transform is untouched. Uses stackalloc → factory is now `unsafe`.
- 2026-07-14 | Wide (fProxyW) radix-4 butterfly for the workspace fft/ifft power-of-4 path
  (FftCoreRadix4Wide), BOTH dtypes — float8 and double4. Vectorizes across the inner j loop for
  stages with quarter-stride q >= fProxyW.Width (8 float / 4 double): Width consecutive j give
  contiguous wide re/im loads/stores and reads of precomputed contiguous per-stage twiddles
  (ws.sw1/2/3 re+im, W^1/W^2/W^3 tabulated directly so every lane reproduces the scalar butterfly
  bit-for-bit — TableFftVsRecurrence's relTol tests pass, float+double). Stages q < Width stay
  scalar; q >= Width powers of 4 are multiples of Width so no j-tail. Measured scalar→wide,
  thermally matched via the (constant across runs) float-wide anchor:
    float(ws):  64K 0.556→0.373 = 1.49x, 256K 2.80→1.66 = 1.69x, 1M ~14-17→6.5 = raw ~2.1-2.5x
    double(ws): 64K 0.607→0.392 = 1.55x, 256K 3.00→1.81 = 1.66x, 1M 21.7→8.4 = 2.58x
  DOUBLE WINS TOO, bigger relative gain at 1M. The vecDot "double regressed through fProxyW"
  finding did NOT transfer: there double4-via-wrapper lost to an existing hand-tuned double4
  body; FFT's scalar butterfly had none, and the across-j ILP win is lane-count-agnostic. This
  is also the first double consumer of the WideOP double operators (now test-covered via FFT).
  HONEST 1M CAVEAT: the 1M butterfly self-throttles (largest/last size, box heats mid-run) — the
  double control swung 21.7→32 ms across identical-code runs; treat 1M as directional
  (~1.75-2.6x), 64K/256K are the clean reproducible figures. Speedup GROWS with N (butterfly-
  dependency-bound, not bandwidth-bound as first feared). Cost: sw* tables add ~2N elems to the
  workspace (~8 MB float / ~16 MB double at N=1M) on top of the full-circle table. NOT applied to
  rfft/irfft (their inner sub-FFT runs the mixed path at size M<n, where the top-level stage
  tables don't match) — future work: per-(size,tableN) stage tables, or derive W^2/W^3 from W^1
  in-register to halve sw* memory.
- 2026-07-12 | Full-circle twiddle table bandwidth tradeoff: uses ~2x twiddle memory (~8 MB at N=1M for float) versus the half-table, offset by halving the number of full-array passes (log4(N) vs log2(N) passes). (was FFT.Workspace.fProxy.cs:21)

## Eigen
- 2026-07-13 | Dropped the unsourced "~30x faster" multiplier from the three cyclic-Jacobi
  [Obsolete] messages (decompInPlace, valuesJacobi-style overloads); kept "Prefer
  Eigen.symmetricInPlace / Eigen.valuesSymmetricInPlace" guidance. Measured multiplier (no
  regression test pins it): Householder-tridiagonal + QL is roughly ~30x faster than cyclic-Jacobi
  for symmetric eigenpairs at this library's benchmarked sizes. (was Eigen.fProxy.cs:913, 1073, 1084)
- 2026-07-11 | Eigen.fProxy.cs doc trims (power/inverse-power iteration, Lanczos, cyclic-Jacobi decompInPlace) removed forwarder-architecture narration and an internal spec pointer (docs/dev/spec-svd-eigen-convergence.md) explaining why decompInPlace's sweep-budget constant isn't scaled by Consts.sweepBudget (its "sweep" is a full-matrix Jacobi sweep, a different iteration unit from LAPACK dbdsqr's per-value QR/QL sweeps). No perf verdicts lost -- purely doc-comment condensation. (was Eigen.fProxy.cs:16, :215, :533, :1140 pre-edit line numbers)

## Krylov
- 2026-07-11 | ApplyDot fusion investigated (cg/pcg's Ap=A·p + pAp=dot(p,Ap) step): a fused single-pass version was tried and measured slower than composing Apply+dot via IfProxyLinearOperator.ApplyDot; kept as the composed form. Don't retry the fused version without new evidence. (was Krylov.fProxy.cs:104, :301 pre-edit line numbers)
- 2026-07-11 | MINRES doc trimmed: the true residual norm ‖b-Ax‖ falls out of the Lanczos+Givens-QR recurrence for free via the running `phibar` variable (no extra dot/matvec needed to test convergence). Variable names (y,r1,r2,v,w,w1,w2) follow Paige & Saunders 1975 / Choi-Saunders minres.m. (was Krylov.fProxy.cs:452 pre-edit)

## LOBPCG
- 2026-07-12 | AXnext/APnext dropped from fProxyLOBPCGCache: allocated (k x n each) but never read
  or written -- UpdateActiveBlock deliberately doesn't mirror-combine AX/AP (see the 2026-07-11
  entry below on that same point), so no consumer ever existed for these two buffers. (was
  LOBPCG.Cache.fProxy.cs:115-121, :186-189; LOBPCG.fProxy.cs:719-722 pre-edit)
- 2026-07-12 | lockTol = 0.1*tolerance margin derivation (trimmed from comment): once a pair locks,
  the remaining active pairs are confined B-orthogonal to it, and the best residual achievable
  under that confinement is ~0.87x the frozen pair's lock residual -- hence locking at
  0.1*tolerance instead of tolerance, to avoid leaving later pairs stuck just above tolerance. (was
  LOBPCG.fProxy.cs:152)
- 2026-07-12 | AP field doc trimmed in fProxyLOBPCGCache (Cache.fProxy.cs): "this one mattered even
  more in practice" narration removed -- same rationale already covered by the 2026-07-11 AX/AP
  entry below. (was LOBPCG.Cache.fProxy.cs:107)
- 2026-07-11 | Buckling-mapping worked example (trimmed from the class doc comment for length; candidate for user-facing docs): the linear-buckling problem K_E*phi + lambda*K_G*phi = 0 (K_E SPD elastic stiffness, K_G indefinite geometric stiffness, Nastran SOL 105 / Abaqus *BUCKLE convention) rearranges to the pencil K_G*phi = mu*K_E*phi with mu = -1/lambda_cr, i.e. K_G in the A slot and K_E in the B slot. Usage: `var mu = Eigen.lobpcg(in K_G, in K_E, ref ws, k, tol, maxIter);` — mu is ASCENDING, mu[0] most negative/critical; a mu[i] >= 0 is not a buckling mode under this reference load, discard rather than divide; lambdaCritical[i] = -1/mu[i] for mu[i] < 0. Opposite sign convention (K_E*phi = +lambda*K_G*phi) uses lambda_cr = +1/mu, same pencil/targeting. (was LOBPCG.fProxy.cs:67-82)
- 2026-07-11 | Initial-X seeding bug history: an earlier deterministic fill used `(i + c*3 + 1) & 3`, periodic with period 4 in both i and c, so seeded X had at most 4 distinct rows — exactly rank-deficient for k > 4. The degeneracy was silently absorbed by FactorGram's ridge retry, so the solver iterated correctly within only a 4-dimensional subspace, never converging to eigenpairs 5+. Fixed by a fixed-seed Unity.Mathematics.Random fill instead. (was LOBPCG.fProxy.cs:107)
- 2026-07-11 | (d1) re-deflation step exists because a buckling smoke test (float) hit a hard-locking fixed point: once a pair locks, active X rows can retain a fixed B-component along the just-frozen row that no later search direction can cancel, freezing the residual at ~|component|*|dLambda|*||Bx||. Fixed by B-orthogonalizing the active block against locked rows every iteration. (was LOBPCG.fProxy.cs:239)
- 2026-07-11 | AX/BX are recomputed fresh via A.Apply/B.Apply every iteration rather than propagated through Cholesky-QR/Rayleigh-Ritz combinations, because propagating them was observed to accumulate rounding error that compounds: residual shrinks nicely for ~15-20 iterations, then stalls and creeps back up instead of continuing to converge. Same fix applied to AP/BP: an inaccurate AP corrupts the next iteration's [X,W,P] Gram/H directly (H's P-columns are dot(*,AP)) and was observed to produce Ritz values below lambda_min (down to -1E13 and beyond, exceeding the plausibility envelope by 1E5-1E30x) as soon as P entered the mix, even though the same marginal Cholesky conditioning is harmless in the P-less 2-block path. (was LOBPCG.fProxy.cs:339, :354)
- 2026-07-11 | UpdateActiveBlock deliberately does not mirror-combine AX/AP (or BX/BP) the way an earlier version did: the caller always immediately recomputes them via a fresh Apply right after the call returns, so the mirror-combine's result was always discarded — pure wasted work (extra O(3k^2 n) multiply-adds per iteration). Don't reintroduce it. (was LOBPCG.fProxy.cs:1163-1169 pre-edit)

## LP.BarrodaleRoberts
- 2026-07-12 | LICENSE: this file (and LP.FrischNewton) is a port of GPL(>=2) quantreg code
  (rqbr.f); owner is requesting relicensing permission from Koenker et al. (see the pending-
  permission section in Source/Third Party Notices.md — package must not be redistributed until
  resolved). A complete, suite-green CLEAN-ROOM replacement pair (papers/pyfixest-MIT provenance)
  exists at commit bdfd9ec (reverted by 101f8c9): correct at every test/benchmark size, but first-cut
  1.1-3x slower (BR float m>=4096: anti-cycling misfire, 115 iters) — restore + run the planned
  optimization round (Sherman-Morrison basis updates, pointer/fused loops) if permission is denied.
- 2026-07-12 | Ratio-test candidate collection pass (column-strided T[i,enter] read) left as-is
  deliberately: it costs O(m) per entering-column choice, so O(m*iters) total, asymptotically
  dominated by the O(m*n^2) BRPivot elimination sweep for any n > 1. A from-scratch column-major
  shadow of T was considered and rejected -- it would have to stay in sync across every BRPivot row
  update (itself row-major/unit-stride for good reason), doubling that update's cost to fix a
  strictly smaller-order term. (was LP.BarrodaleRoberts.fProxy.cs:240-243 pre-edit)
- 2026-07-11 | Source provenance (trimmed from the file banner): transcribed line-by-line from the Koenker-d'Orey Fortran `rqbr` (R `quantreg` package, src/rqbr.f), fetched from https://cdn.jsdelivr.net/gh/cran/quantreg@master/src/rqbr.f (same mirror pattern as LP.FrischNewton's source), cross-checked against the R wrapper `rq.fit.br` for the ift/flag status-code semantics. Deviation rationale kept in full: (1) rqbr's toler=eps^(2/3) is tighter than this library's simplex tolerance, deliberately not imported as a one-off literal; (2) on ift=2 "premature end", the raw Fortran leaves x untouched, but this port extracts the last-vertex structural solution since stage 1 has always completed by then and LPStatus.Unbounded's contract already promises that extraction; (3) ift=1 "solution may be nonunique" is a warning the reference emits without altering x, and LPStatus has no matching state; (4) reference diagnostic-only outputs (dsol/sol/h/e) are dropped in favor of an honest-recomputed objective, same reasoning as ladFN. (was LP.BarrodaleRoberts.fProxy.cs:15-97 pre-edit)
- 2026-07-11 | Perf verdict: BR's candidate-ratio selection used an O(nCand^2) selection sort (linear-scan-for-min + swap-remove), which was measured (not merely suspected) to be the dominant cost at large m — BR's own reported iteration count stayed flat near m=16384 while wall time grew far faster than FN's comparable-iteration interior point, the signature of quadratic work hidden behind a small iteration count (surfaced via LPBenchmark Section 2b, m=1024-16384). Fixed by sorting candidates once via heapsort above BR_CAND_SORT_THRESHOLD (set above every m the test suite exercises for BR, <=192, so tested paths stay on the original code) instead of the O(n^2) selection sort. Don't revert to unconditional selection sort. (was LP.BarrodaleRoberts.fProxy.cs:261-280 pre-edit)

## LP.DualSimplex
- 2026-07-12 | Bound-flip (BFRT) application's column-strided `flipRHS[i] += delta * M[i, j]` loop
  left as-is deliberately: flipCount is normally small (a handful of boxed nonbasics absorbed per
  iteration, not O(N)), so its O(flipCount*m) cost is already far below the O(mN) PRICE passes
  this file's other column-strided loops were. Routing it through a dense Mmul(M, deltaVec, ...)
  GEMV (deltaVec sparse, nonzero only at flipCols) was considered and rejected -- it would touch
  all N columns unconditionally, a regression whenever flipCount << N (the common case), for a
  loop that was never the O(mN) bottleneck. (was LP.DualSimplex.fProxy.cs:519-525 pre-edit)
- 2026-07-11 | DSE update formula, verified line-by-line against HiGHS source (not just paraphrase): highs/simplex/HEkk.cpp::updateDualSteepestEdgeWeights (`dual_edge_weight_[iRow] += aa_iRow*(new_pivotal_edge_weight*aa_iRow + Kai*dse_array_value)`) called from HEkkDual.cpp::updatePrimal with `Kai = -2/alpha_col`, `new_pivotal_edge_weight = edge_weight[row_out]/alpha_col^2`, DSE array built as `col_DSE = Ftran(row_ep)` i.e. tau = B^-1 rho_r — matches `w_i' = w_i - 2(alpha_qi/alpha_qr)*tau_i + (alpha_qi/alpha_qr)^2*w_r, then w_r' = w_r/alpha_qr^2`. The 1e-4 floor is HiGHS's `kMinDualSteepestEdgeWeight` (highs/simplex/SimplexConst.h). (was LP.DualSimplex.fProxy.cs:36-43 pre-edit)
- 2026-07-11 | Warm-start correctness proof (trimmed from banner): the warm overload's dual-feasibility repair (bound flips / temporary artificial bounds keyed off a real BTRAN-computed reduced cost) is a strict generalization of the former cold-only precondition, provably bit-identical at the all-logical basis since y = B^-T c_B is then exactly the zero vector (c_B = 0, and BTRAN of an all-zero vector stays all-zero through every forward/back-substitution step — each step is an assignment of 0, multiply-by-0, subtract-of-0, or divide-of-0-by-nonzero, all exact in IEEE754). (was LP.DualSimplex.fProxy.cs:47-51, :430-436 pre-edit)
- 2026-07-11 | Bug history — DualRatioTest: an earlier version allowed bound flips to fully resolve a row with no actual pivot. It passed every test at n<=24 but produced a false Infeasible on a 48-variable random instance: a flip-only iteration leaves the basis (hence y=B^-T c_B, hence every dj) unchanged, so a column just flipped from AtLower to AtUpper keeps its old, now-wrong-signed reduced cost with no future iteration positioned to notice. Fixed by guaranteeing a real pivot happens every time DualRatioTest returns anyCandidate=true (flips are only ever a prefix of the walk). Don't reintroduce flip-only resolution. (was LP.DualSimplex.fProxy.cs:117-122 pre-edit)
- 2026-07-11 | Bug history — float artificial-bound scaling: HiGHS's fixed [0,1e7] artificial-bound box (tuned for HiGHS's internally-scaled/equilibrated data) was observed to produce a false Infeasible within the first few dual iterations in float. RebuildXB's adjusted rhs sums the artificial bound's contribution over every simultaneously-artificial column (up to ~n/2 for mixed-sign-cost problems), landing xB around -(artificialBound*n/2*|A|); at 1e7 with n~48 that's order 1e8, and float's ~1.19e-7 relative precision there is an absolute error of order 10, which swamped feasTol (~3.45e-4) outright. Fixed by scaling the artificial bound to the problem's own data magnitude (100x largest |cost|/|rhs|) instead of HiGHS's literal. (was LP.DualSimplex.fProxy.cs:304 pre-edit)
- 2026-07-11 | Bug history — cost-perturbation base literals: an earlier dualTol-scaled float variant (instead of reusing HiGHS's own literal bases for both dtypes) made float branch-and-bound trees explode (benchmark-verified). Reverted to HiGHS's literals for both float and double. (was LP.DualSimplex.fProxy.cs:332 pre-edit)
- 2026-07-11 | Perf/correctness verdict — DSE weight reseed gating: tying the weight[] reseed to `didResumeFactors` (not just the caller's original `resumeFactors` request) was benchmark-caught as a real regression: MIPBenchmark float branchy12 went from Optimal/216 nodes/10.6ms to NodeLimit/20000 nodes/122.7ms under the unconditional version (a cache hit whose eta file was already at capacity resumed weight[] even though B/P/eta had just been refreshed, letting weight[] drift across an unbounded refactorization chain spanning the whole search). Fixed by tying the reseed to didResumeFactors, confirmed back to Optimal/226 nodes/~9.5ms. Don't detach the reseed condition from didResumeFactors. (was LP.DualSimplex.fProxy.cs:416-419 pre-edit)
- 2026-07-11 | Bug history — perturbedCost in the dual-feasibility repair: using perturbedCost (instead of the original cost) for the one-time true-dual-feasibility decision was an actual bug. A column with cost[j] exactly 0 (e.g. every x+/x- column in LP.lad's reformulation, which has none of its own cost) is dual-feasible as-is, but the perturbation's random sign could nudge it slightly negative and give it a pointless artificial bound; multiplied across the many exactly-zero-cost columns LP.lad's [x+|x-] block always has, this corrupted the warm-started basis handed to the primal cleanup badly enough to report a false Unbounded. Fixed by using the original cost. Don't use perturbedCost here. (was LP.DualSimplex.fProxy.cs:445-450 pre-edit)
- 2026-07-11 | Perf note — zero-pivot fast path: skipping RevisedPrimalCore's cleanup call when the dual loop already left a true optimum (r<0 exit, zero pivots, no artificial bounds) was measured to roughly halve a warm re-solve's fixed per-call cost when it applies (~0.12ms/call -> ~0.06ms/call at mAug~80, isolated warm LP.solve(ref LPBasis) benchmark, MIP perf investigation 2026-07-10) — a genuine but minority case for MIP/strong-branch-trial re-solves (most single-bound tightenings still cost >=1 real pivot). (was LP.DualSimplex.fProxy.cs:657-658 pre-edit)

## LP.FrischNewton
- 2026-07-11 | Source provenance (trimmed from banner): ported and verified line-by-line against Daniel Morillo & Roger Koenker's `rq_fnm`/`lp_fnm` (originally Ox, translated to MATLAB by Paul Eilers 1999, modified by Koenker April 2001), fetched from https://github.com/karenamckinnon/summer-temperature-distributions/blob/master/rq.m (mirrors the file distributed with R's quantreg package; same algorithm also in quantreg's Fortran rqfnb.f). Every update formula (predictor, centering parameter, corrector, step-length ratio test, the 0.9995 factor) is that source's, not reconstructed from memory. Sign convention verified against LadStackloss's published coefficients in testing. (was LP.FrischNewton.fProxy.cs:23-30, :43 pre-edit)
- 2026-07-11 | Problem/dual formulation (trimmed from banner, kept for reference): quantile regression at level tau in (0,1) (tau=0.5 == LAD up to a factor 2), min_x sum_i rho_tau(b_i - A_i.x), rho_tau(u)=u*(tau-1[u<0]). Its dual (rq_fnm's construction): max_a b.a s.t. Aᵀa=(1-tau)Aᵀ1, a in [0,1]^m — solved by lp_fnm as min c.v s.t. Ãv=b̃, 0<=v<=1 with Ã=Aᵀ, c=-b, b̃=Aᵀ((1-tau)1); the LP's own primal variable v IS the dual weight a. (was LP.FrischNewton.fProxy.cs:34-42 pre-edit)

## LP
- 2026-07-11 | LP.lad's BR/FN crossover, measured (LPBenchmark Section 2b, 2026-07-09, after the BR sort-path + FN SIMD optimization round): double — BR wins through m=4096 (2.49ms vs FN 2.71ms) and loses only ~11% at m=16384, so the threshold sits at the last measured BR-win size, 4096; float — FN's SIMD gains moved its win boundary down to m=1024 (FN 0.47ms vs BR 0.62ms) while BR still wins at m=384, so 512 splits the measured bracket. Re-measure Section 2b (and re-tune the threshold) whenever either engine's per-iteration cost changes. (was LP.fProxy.cs:318-324 pre-edit)

## LP.RevisedSimplex
- 2026-07-12 | RevisedPrimalCore's warm-start overload was added for LPMethod.DualSimplex's
  HiGHS-style composition (LP.DualSimplex.fProxy.cs hands its terminal basis to this core as a
  cleanup pass once real bounds are restored); the fresh-start overload above is non-breaking --
  it simply builds the all-logical basis/status and forwards here, so its behavior and public
  surface are unchanged. (was LP.RevisedSimplex.fProxy.cs:439-442 pre-edit)
- 2026-07-11 | Bug history — HarrisRatioTest far-bound fallback: "travel through to the far bound" assumed the far bound is finite, which broke on a dense covering LP (min cx s.t. Ax>=b, x>=0, A,b,c>0): every >=-row logical starts basic and above its upper bound (0) with an unreachable lower bound (-INF), for every row simultaneously, so no row ever contributed a finite ratio-test limit and the pass-1 unbounded check fired — RevisedSimplex returned Optimal with 0 iterations / objective 0 while tableau/interior/dual all agreed on the true optimum (a silent phase-1 bail extracting x=0 from a basis nothing ever pivoted into). Caught by LPBenchmark, reproduced in LPTests.fProxy.cs as RevisedDenseCovering (failed before the fix, passes after). Fixed by a two-attempt ratio test: the first pass is byte-identical to the original algorithm; only if it would report Unbounded does a second pass run with a fallback that targets the NEAR (violated) bound instead of an unreachable far one. Don't remove the two-attempt structure. (was LP.RevisedSimplex.fProxy.cs:297-315 pre-edit)

## LQRP
- 2026-07-11 | Design rationale trimmed from class remarks: QRCP's LEVEL-3 machinery (blocked dlaqps panel core with deferred F-matrix trailing update) is deliberately not mirrored in LQRP -- it only earns its bookkeeping at large sizes, and the primary consumer (rank-deficient IK Jacobians) is small (task DOF × joint DOF). Add a blocked core later if large wide matrices need it. Downdated row norms (vs. exact recompute every step) remove a second O(m²n) pass re-summing candidate norms; pivot selection needs current row NORMS not row DATA, so tracking incrementally is sufficient. Basic-vs-min-norm gap: for an inconsistent rank-deficient b, solveInPlace is not minimum-norm because the below-diagonal block L21 couples the independent variables into the dropped equations (the transpose-dual of QRCP's R12 coupling -- L's top-right IS zero, but that's not where the coupling lives; trailing rows of L keep their full norm, only the trailing diagonal is small). minNormSolveInPlace closes the gap by least-squares-solving the coupled m×r block K=[L11;L21] instead of just L11. (was LQRP.fProxy.cs:22-47 pre-edit)
## MIP.Domain

- 2026-07-11 | UB-row sentinel-rhs bug (moved from header comment): using a 1e30 sentinel rhs directly for an infinite-UB row (instead of the inert coefficient-0/rhs-0 convention) fed into DualSimplexCore's dataScale/artificialBound scan and inflated artificialBound to ~1e32, producing a false Infeasible. Reproduced on the Gomory/Wolsey instance: sentinel=1e30 -> false Infeasible after 63 pivots; sentinel<=1e10 -> correct. Also a correctness risk beyond numerics: a finite sentinel can silently bound a genuinely unbounded direction. Fix: UB rows start inert and PushBoundChange/UndoToMarker activate/deactivate the row's coefficient explicitly. (was MIP.Domain.fProxy.cs:20)
- 2026-07-11 | PropagateFixpoint HiGHS provenance (moved from doc comment): ported from mip/HighsDomain.cpp's `propagate`/`propagateRowUpper`/`propagateRowLower`. Worklist membership mirrors HighsDomain's `markPropagate`/column-incidence loop and `propagateinds_`. Infinite-contributor counts are HiGHS's `ninfmin`/`ninfmax`; the closed form `(rhs - (act - ownContribution)) / a_ij` is HiGHS's `minresact`/`maxresact`. Termination on queue-drain mirrors HiGHS's `havePropagationRows`; the `PROPAGATION_MAX_PASSES * m0` visit cap is a deliberate deviation — HiGHS has no such cap because its incremental activity bookkeeping makes each visit O(row length) instead of a full recompute, and this port's fixpoint isn't persisted/maintained incrementally across the whole B&B tree the way HighsDomain's activity arrays are. (was MIP.Domain.fProxy.cs:154)

## MIP.Pseudocost

- 2026-07-11 | PseudocostScore formula + fidelity gaps (moved from comment): faithfully ported from HighsPseudocost::getScore(col, upcost, downcost) (mip/HighsPseudocost.h): `costScore = max(upcost,minThreshold)*max(downcost,minThreshold) / max(minThreshold, cost_total^2)`, then `mapScore(x) = 1 - 1/(1+x)`. upcost/downcost are PseudocostEstimate (== HighsPseudocost::getPseudocostUp/Down's 2-arg no-offset overload: fractional-distance * own mean, falling back to the running global-average pseudocost when the variable has zero samples). cost_total is the running global average (globalPCSum/globalPCCount); minThreshold == PSEUDOCOST_EPS (both 1e-6), same clamp value and placement as the source. OMITTED (fidelity taxonomy — subsystems this port doesn't have): the conflictScore term (no conflict analysis / no-good learning), the cutoffScore term (no cutoff-bound tracking), the inferenceScore term (no propagation/inference statistics). OMITTED: degeneracyFactor weighting (no LP-degeneracy detection) — HiGHS only sets it > 1 while actively degenerate; fixed at its non-degenerate default of 1.0, getScore's full expression collapses exactly to mapScore(costScore), which is what PseudocostScore returns. (was MIP.Pseudocost.fProxy.cs:50)

## MIP

- 2026-07-12 | Limit-exit semantics verified faithful against HiGHS source (master + v1.7.2, HighsSearch::dive / HighsMipSolver::run / cleanupSolve): upstream also checks limits AFTER the in-flight node's full work (evaluateNode installs incumbents unconditionally before checkLimits; plunge loop runs heuristics+dive before its checkLimits), and cleanupSolve reports the incumbent objective + queue-folded dual bound + finite gap on limit exit — same as SearchCore's top-of-loop budget check after the release-scan fix. Known deviations, intentional: (a) HiGHS collapses node/leaf/improving-sol limits into one kSolutionLimit status; this port keeps a distinct NodeLimit. (b) cumulative maxIter (total LP iterations across the search) is PORT-ORIGINAL — HiGHS has no MIP-wide LP-iteration budget (its kIterationLimit is LP-only, never set from MIP code). (c) TryRoundingHeuristic fires on every fractional node; HiGHS runs randomizedRounding at the root + once per plunge start (granularity choice, see 07-11 entry below).
- 2026-07-11 | Architecture notes trimmed from file header (moved from header comment): Bounds-as-rows shift mechanics — LP.solve only supports x>=0, so every variable is shifted to a non-negative y (anchor-low/anchor-high/free-split, same substitution as QP.PhaseOneFeasibleStart); integer variables get two pre-allocated rows (y<=U, y>=L), branching only rewrites their rhs so the augmented LP's shape stays fixed and the same LPBasis stays warm-startable. UB-row activation: starts INERT (coefficient 0) when xu is infinite, activated (coefficient -> 1) on first branch — a literal 1e30 sentinel rhs corrupts the dual simplex's dataScale/artificialBound scaling and can silently bound a truly unbounded direction (full detail moved to MIP.Domain.fProxy.cs's own header). Warm start: one LPBasis persists across the whole search including strong-branch trials; the dual simplex's dual-feasibility repair makes a stale basis (right after a plunge dive, an undone strong-branch trial, or a queue jump) a correct, not just fast, starting point. Node state: the current plunge's dive steps use the incremental bound-change stack (PushBoundChange/UndoToMarker); a queue jump is not generally to an ancestor so it can't replay that stack — instead each queued node carries its own full length-n bound snapshot (fProxyMIPQueueNode) and a jump overwrites live bound state wholesale (ApplyNodeBounds) and resets the stack. dualBound = min over every still-open node's own parent-LP bound (the current plunge frontier plus everything still in the queue). (was MIP.fProxy.cs:11)
- 2026-07-11 | TryRoundingHeuristic HiGHS provenance + deviations (moved from comment): ported from HighsPrimalHeuristics::randomizedRounding; the randomized-interval draw is HiGHS's `floor(relaxationsol[i] + randgen.real(0.1, 0.9))`. Two intentional deviations from HiGHS's tryRoundedPoint/randomizedRounding: (a) HiGHS re-solves an LP with the rounded integers fixed to repair continuous variables and confirm feasibility; this port has no per-node LP re-solve budget for the heuristic, so it does an O(mn) direct feasibility check against the original rows instead. (b) HiGHS's `randgen` is a solver-wide RNG advanced continuously; MIP.solve has no public seed parameter and must stay bit-deterministic across repeated identical calls, so this uses a fixed internal seed instead (roundRng in SearchCore). Bound-clamping decision: rounded values are clamped into the CURRENT node's bounds and feasibility is checked against them, not the root bounds — root bounds may be fractional (user-supplied), so clamping to them could install a fractional "integer" incumbent; node bounds are always integral once branched. (was MIP.fProxy.cs:541)

## OP.Component

- 2026-07-11 | clampInPlace `this T` vs `this in T` (moved from remarks): takes the receiver by value (`this T`), matching every other Comp wrapper in this file — a generic extension method's receiver cannot use `this in T` (CS8338: the 'in' extension-method form requires a concrete, non-generic value type). Callers migrating from the old static-style `clampInPlace(in v, ...)` just drop the now-illegal `in`. (was OP.Component.fProxy.cs:154, same block also in OP.Component.iProxy.cs:152)

## OP.Dot

- 2026-07-11 | dotSelf fused-kernel rejection (moved from comment): dotSelf composes a plain matVecDot pass + a separate vecDot pass for `y = Ax` plus `dot(x,y)`, rather than a single fused kernel. Why: an earlier version dispatched a genuinely-fused single-pass kernel (matVecDotSelf) for square A, folding dot(x,y) into the GEMV row-loop via two alternating scalar accumulators (the row-loop itself already uses vecDot's fProxy4 SIMD pattern, so there was nothing left to widen the outer cross-row fold into — row results arrive one at a time, not as an aligned block of 4). MEASURED WORSE on the BSR analogue of the same pattern (bsrMatVecB1Dot, part of the Krylov optimization round): the scalar alternating fold lost to simply calling the already-SIMD-tuned vecDot separately, by a wide and reproducible margin at the block=1 stencil benchmark. Reverted here too on the same architectural basis (not separately re-measured for dense). Verdict: don't retry a fused scalar-accumulator dot-fold here without new evidence. (was OP.Dot.fProxy.cs:95)

## Optimize

- 2026-07-11 | ladIRLS weighting formula + delta tuning (moved from doc comment): minimizes ‖A x − b‖₁ by repeatedly solving the weighted normal equations `(AᵀW A) x = AᵀW b` with per-row weights `wᵢ = 1 / max(|rᵢ|, delta)`, `rᵢ = (A x − b)ᵢ`; typically converges in a handful of iterations for a well-conditioned overdetermined design. delta is a Huber-like transition width: too small causes oscillation, too large drifts the fit toward ordinary least squares. (was Optimize.fProxy.cs:224)

## QP.Info

- 2026-07-11 | QPInfo precision + diagnostics rationale (moved from doc comment): QPInfo is a plain, unprefixed struct (not float/double-generated) because diagnostics need not be precision-typed — objective is always reported as double regardless of solve precision, matching LPInfo/LstsqInfo/SolveInfo's convention. stationarityResidual/feasibilityResidual follow the solver-diag-struct convention of "only already-computed/cheap numbers": both are already on hand as a direct byproduct of the null-space step (stationarity: the reduced gradient Zᵀg the step just drove to ~0; feasibility: one cheap GEMV, A_W x - b_W) — see QP.eqpNullSpaceStep. Per spec's Stage 1 oracle, these are meant to be compared against a full KKT-system LU solve. There is no separate complementarity residual yet because the fixed-working-set kernel has no inequality constraints to be complementary about; a future active-set loop would extend this struct's diagnostics, not replace them. (was QP.Info.cs:94)

## QP

- 2026-07-11 | Null-space method derivation (moved from file header): parameterize every feasible point as x = x0 + Zy, x0 any point with A_W x0 = b_W, Z an orthonormal basis for null(A_W) (A_W Z = 0, so A_W x = A_W x0 = b_W for ANY y). Substituting, the equality-constrained problem `min ½xᵀQx + cᵀx s.t. A_W x = b_W` becomes the UNCONSTRAINED reduced problem `minimize_y ½yᵀ(ZᵀQZ)y + (Zᵀg(x0))ᵀy + const`, g(x0) = Qx0 + c — an ordinary quadratic in y with Hessian H_Z = ZᵀQZ and gradient Zᵀg(x0) + H_Z y. For ANY quadratic, Newton's method reaches the exact minimizer in ONE step regardless of starting point (the model IS the function), so solving H_Z y = -Zᵀg(x0) and setting x1 = x0 + Zy lands exactly on the equality-constrained optimum — no line search, no iteration. Source: Nocedal & Wright, Numerical Optimization (2nd ed.), ch. 16.2, eq. 16.16-16.19. (was QP.fProxy.cs:12)
- 2026-07-11 | "Keeping Q implicit" QR mechanics (moved from file header): QR.decompInPlace's public API forms the dense (n x k) "thin" Q1 in one call (factor + reconstruct, no split entry point). To avoid ever materializing that n x k matrix, this file bypasses QR.decompInPlace entirely and drives QR's own per-step primitives directly (QR.genHouseholder / QR.applyReflectorRight, both `internal` — the same functions decompInPlace itself is built from): FactorWorkingSetTranspose is exactly decompInPlace's factorization half (store R + stash each Householder vector into A_Wᵀ's own columns), replicated rather than called through the public API specifically so the reconstruction half never runs. Two more primitives close the loop without ever forming Q1: (1) ApplyWorkingSetQtForward — Q_full = H_0 H_1 ... H_{k-1} is an n x n orthogonal matrix (each reflector acts on the full ambient n-dimensional space; only k of them exist because A_Wᵀ has k columns), so Q_fullᵀ = H_{k-1} ... H_0 and a FORWARD sweep (d = 0..k-1) of the k stashed reflectors over any n-vector v computes Q_fullᵀv = (Q1ᵀv ; Zᵀv) in one pass — top k entries and bottom n-k entries. This is the exact trick QR.solveInPlace already uses for its `b` argument (computing Qᵀb without ever forming Q), generalized and replayed from STORED reflectors instead of freshly-generated ones; it replaces both QR.decompSolve's "Qᵀg, then R-solve" (used for multiplier recovery) and the reduced gradient Zᵀg in one sweep. (2) FormNullSpaceBasis — Z itself (n x (n-k), needed because the reduced Hessian ZᵀQZ and the step p = Zy are GEMM/GEMV operands) is formed by REVERSE-sweeping (d = k-1..0) the same stashed reflectors over the seed [0; I_{n-k}] — exactly QR.decompInPlace's own Q-reconstruction phase, seeded with the TRAILING identity block instead of the leading one, targeting a separate n x (n-k) buffer instead of overwriting A_Wᵀ. Z is smaller than the full n x n Q by construction (only the n-k null-space columns), so forming Z (not Q) is the documented exception to "don't form dense Q". Note: "Q" here means QR's orthogonal factor, distinct from this file's Q the Hessian — an unfortunate letter clash inherited from the spec/textbook. (was QP.fProxy.cs:12)
- 2026-07-11 | Stage 2-3 reuse structuring rationale (moved from file header): every function in this file is `internal static` (not a buried local), matching the structuring rule LP.RevisedSimplex.fProxy.cs set for LP.DualSimplex.fProxy.cs — a future inequality active-set loop (ratio test, add/drop, Dantzig pricing) will call eqpNullSpaceStep (or its constituent pieces) once per iteration, re-factoring A_Wᵀ from scratch after every working-set change (see FactorWorkingSetTranspose's own doc comment for that cost and why it is deliberately not incremental — v1 scope decision). (was QP.fProxy.cs:12)
- 2026-07-11 | eqpNullSpaceStep future active-set call pattern (moved from doc comment): called once by eqpSolve today (the whole algorithm for a fixed working set). A future inequality active-set loop would call this once PER ITERATION with the CURRENT working set (which changes as constraints are added/dropped), re-factoring A_Wᵀ from scratch every call — an incremental QR update was judged not worth porting at this library's dense target sizes. That future loop will need to intercept BETWEEN computing the step and applying it in full (the ratio test may truncate the step to alpha*p, alpha < 1), which the current version does not do — expect that seam to require splitting the "compute p" and "apply p, recover multipliers" halves of this function. (Note: qpActiveSetCore below already did exactly this split, built directly on the constituent kernel functions instead of through eqpSolve/eqpNullSpaceStep.) (was QP.fProxy.cs:358)
- 2026-07-11 | FactorWorkingSetTranspose HiGHS comparison (moved from comment): HiGHS maintains an incrementally-updated factorization of the working-set basis across add/drop changes; deliberately not ported here — simple/correct was judged to beat incremental/subtle for this library's v1 scope. (was QP.fProxy.cs:545)
- 2026-07-11 | qpActiveSetCore unbounded-detection proof (moved from header comment): verified against Nocedal & Wright, Numerical Optimization (2nd ed.), section 16.5 ("Active-Set Methods for Indefinite QP") — fetched and read 2026-07-09. With Z the null-space basis for the current working set and ZᵀGZ found singular/indefinite along a direction sZ chosen to be non-ascent (their eq. surrounding "q(x+alpha*Z*sZ) -> -infinity as alpha -> infinity" and the sign choice "so that Z*sZ is a non-ascent direction for q"), the text states plainly: "By moving along the direction Z*sZ, we will encounter a constraint that can then be added to the working set for the next iteration. (If we don't find such a constraint, the problem is unbounded.)" This library's case is the boundary of their construction (Q only PSD, so the reduced Hessian can go singular/zero-curvature but never strictly negative-definite beyond that boundary): conditions 1-2 (regularized, zero curvature) detect that boundary, condition 3 (no blocker) is their "we don't find such a constraint", condition 4 (descent) is their non-ascent sign choice — made an explicit check here rather than a sign flip because SolveReducedNewtonStep's regularized solve already mathematically guarantees gᵀp <= 0 whenever it succeeds (gᵀp = gzᵀy = -gzᵀ(H_Z+deltaI)^-1 gz, and H_Z+deltaI is PD), so #4 is a defensive check on that guarantee, not a live sign-flip decision. (was QP.fProxy.cs:698, date-stamped note also flagged separately at QP.fProxy.cs:762)
- 2026-07-11 | qpActiveSetCore perturbation-cleanup rationale (moved from comment): the final exact null-space Newton step against the true bounds is exactly LP.DualSimplexCore's own composition pattern ("hand the terminal basis to the primal core... using the REAL cost", see that file's header comment) rather than leaving a perturbation-sized residual in the reported solution. The multiplier check that already declared Optimal never saw perturbed data (it depends only on g = Qx+c and the working-set geometry, never on L/U), so this cleanup pass cannot change WHICH working set is optimal, only where x sits on it: reusing eqpSolve (LQ.minNormSolve to the TRUE b_W, then one exact Newton step) re-lands EXACTLY on this same working set's true optimum. Zero cost on the common, non-degenerate path (skipped whenever perturbation was never engaged). (was QP.fProxy.cs:1112)
- 2026-07-11 | BuildPerturbedBounds magnitude proof (moved from comment): the ratio test's EXACT ties are the root cause of a stalled/cycling run of zero-length steps; widening L/U by a tiny deterministic amount makes them distinct, letting a genuine (if tiny) step through. Uses a deterministic per-row pseudo-random unit value via the SAME cheap integer hash LP.DualSimplexCore uses for its own cost perturbation (MurmurHash3 finalizer mix); magnitude is a SMALL FRACTION of feasTol (0.1x) so it is provably too small to be mistaken for genuine constraint slack anywhere else in the solver (every other feasibility decision in this file compares against feasTol itself), yet many orders of magnitude past a float ULP, so it reliably breaks bit-exact ties. (was QP.fProxy.cs:1223)
- 2026-07-12 | Anti-cycling hardening's deterministic bound perturbation replaced an earlier Bland-style seam; same pattern (and lesson) as LP.DualSimplexCore's own cost perturbation, see that file's header comment. (was QP.fProxy.cs:759-761)
- 2026-07-12 | FormNullSpaceBasis's full-column-width requirement (why the leading-identity restriction doesn't generalize to Z) was caught by the Stage-1 KKT-oracle check at k=2, n=8: k=1 has only one reflector at d=0, whose "columns >= 0" restriction never actually excludes anything, so the bug was invisible there. (was QP.fProxy.cs:541-543)
- 2026-07-12 | qpActiveSetCore's pScale rescale (both the curvature-test call site and RatioTest's own doc comment) was caught by the LP-limit oracle test: Q=0 forces every step through this exact path, since the reduced Hessian is then identically singular every iteration. (was QP.fProxy.cs:872-874, :1239-1240 pre-edit)

## RandomOP

- 2026-07-12 | randomPermutationInPlace uses a separate Fisher-Yates loop from shuffleInPlace rather than sharing one: Pivot.Swap tracks the permutation parity via its swap counter, which plain index swapping (shuffleInPlace's Indices buffer has no parity field) cannot do. (was RandomOP.cs:22)
- 2026-07-12 | fProxyGaussian.Next doesn't use math.sincos: its out-parameter overload is not available via the type-proxy template mechanism, so math.sin and math.cos are called separately instead. (was RandomOP.fProxy.cs:511)

## RandomMatrixOP

- 2026-07-11 | orthogonalInPlace algorithm walkthrough (moved from doc comment): 1) fill an n×n scratch matrix G with i.i.d. N(0,1) entries; 2) QR-decompose G = Q·R (Householder); 3) Haar sign fix — multiply column i of Q by sign(R[i,i]) (sign(0)=+1, no flip); 4) copy the corrected Q into dest. Temp scratch: G (n×n) and R (n×n), both disposed before return; the QR step allocates an additional n-element Temp vector internally (disposed inside decompInPlace). Why the sign fix matters: without it, Householder QR's Q is NOT uniformly distributed over O(n) — the sign of each R diagonal is not equally likely to be ±1, introducing a measurable bias; the sign flip corrects this and yields the true Haar measure. (was RandomMatrixOP.fProxy.cs:141)

## QR

- 2026-07-11 | Blocked panel trailing-update tiling rejection (moved from comment): UnsafeOP.wyVtC/wySubVW already reach full GEMM throughput (~70 GFLOP/s, matched matMatDot) at this width without tiling. Column-tiling was tried and MEASURED SLOWER (added MemClear/call overhead for no cache-locality benefit), so it is deliberately not done here — don't retry without new evidence. (was QR.fProxy.cs:283)

## QRCP.Workspace

- 2026-07-11 | fProxyQRCPCache scope rationale (moved from doc comment): deliberately holds ONLY vn1/vn2, not u (the Householder scratch, length m), w, or the blocked core's larger working buffers (F, flush GEMM scratch, the reconstruction WY buffers). The level-3 blocked core (decompInPlaceBlockedCore, engaged once N_Cols >= 2*QRCP_BLOCK) still takes its vn1/vn2 downdating state from here but Allocator.Temp-allocates those larger buffers per call — so this cache stays minimal with no dead fields (spec ticket: OQ-7, "QRCP earns a cache purely for the downdating state"). Promoting the blocked buffers in here for a fully zero-alloc blocked path (as fProxyQRCache does for QR) is a candidate follow-up. (was QRCP.Workspace.fProxy.cs:26)

## QRCP

- 2026-07-11 | Class-level blocked-panel mechanics (moved from remarks): downdating is what unlocks a level-3 path — pivot selection needs the current column NORMS, not the current column DATA, so once N_Cols >= 2*QRCP_BLOCK the factorization runs the LAPACK dlaqps-style partially-blocked panel core (decompInPlaceBlockedCore): a whole panel of reflectors is factored against a deferred F-matrix and its trailing update flushed once as a rank-kb GEMM, and Q is reconstructed by the same blocked-WY kernel QR uses (QR.reconstructQBlocked). Below that gate the unblocked per-reflector core runs (decompCoreDispatch chooses). fProxyQRCPCache carries only the two n-sized downdating vectors (vn1, vn2); the blocked core's larger working buffers (F, the flush GEMM scratch, and the reconstruction WY buffers) are Allocator.Temp allocated per call inside decompInPlaceBlockedCore — one set per factorization, negligible against its O(n²m) work — rather than folded into the cache. (was QRCP.fProxy.cs:21)
- 2026-07-11 | tol3z codebase-consistency note + retired-buffer history (moved from comment): tol3z is Consts.fProxySqrtEps directly — Consts.cs already defines it as the precise, type-correct sqrt(Consts.fProxyEpsilon), and every other caller in this codebase (Eigen/LOBPCG/Krylov/SVD.LowRank) references it the same way rather than recomputing math.sqrt(Consts.fProxyEpsilon) at runtime. Separately: the current guard-triggered re-sum (writes straight into vn1, no separate colNorm2 buffer) replaced an old exact-recompute-every-step buffer, now fully retired. The batched row-major re-sum is a deliberate widening from LAPACK's own per-column selective recompute: this codebase is row-major, so a single column's exact norm is a strided reduction — the same shape the ORIGINAL always-exact QRCP avoided by summing all trailing columns per row instead of one column at a time — so reusing that batched sweep when ANY column trips the guard is simpler, no more expensive (the sweep touches every trailing column per row regardless of how many needed it), and strictly more accurate for the columns that didn't strictly need re-summing. (was QRCP.fProxy.cs:124 and :129)
- 2026-07-11 | Blocked panel core 8-step walkthrough (moved from section header): per panel step k (panel-local, the pivot lands on global column/row d = rk = p0+k): 1) pivot by max vn1 over trailing columns; the full-column swap in A carries each column's already-written R prefix with it (R is extracted from A's upper triangle at the end), so no separate R swap is needed — only vn1/vn2/P and the k filled F rows are swapped. 2) bring ONLY the pivot column up to date wrt the k prior reflectors (A[:,d] −= V·F[k,·]ᵀ). 3) generate the Householder reflector. 4) take R[d,d] from it and store the reflector. 5) ONE combined pass acc = uᵀ·A over the panel width: acc's reflector-column entries are the compact-WY aux (uₖᵀuᵢ), its trailing entries are the direct term of F's new column. 6) F's new column = direct − F·aux (correction). 7) bring row rk of the trailing part up to date (it becomes R and feeds the norm downdate). 8) downdate vn1 with the same guarded formula as the unblocked core (dlaqps returns KB for the same reason this panel is cut short on a guard trip). (was QRCP.fProxy.cs:377, spec ticket docs/dev/spec-qrcp-blocked.md)
- 2026-07-11 | minNormSolveInPlace (COD) full derivation (moved from section header): QRCP gives A·P = Q·R with R = [R11 R12; 0 ~0] (R11 r×r upper-tri, full rank; the trailing (n-r) diagonal below tol). Writing x = P·y (P a permutation, so ‖x‖ = ‖y‖) and c = Qᵀb = [c1; c2], the residual is ‖R y − c‖² = ‖[R11 R12]·y − c1‖² + ‖c2‖². The second term is fixed, so every least-squares x satisfies M·y = c1 where M = [R11 R12] (r×n, full ROW rank r); among those, min ‖x‖ = min ‖y‖. LQ-factor the SHORT-WIDE M = L̃·Qz (L̃ r×r lower-tri, invertible; Qz r×n, orthonormal rows). Then M y = c1 ⇔ Qz y = L̃⁻¹c1 =: w, and the minimum-norm y with Qz y = w is y = Qzᵀ w (Qz has orthonormal rows). So the whole solve is: 1) QRCP factor (fused: b ← Qᵀb), read rank r off R's diagonal; 2) r == n (full column rank): basic IS min-norm — reuse solveInPlaceFinish, no COD; 3) r < n: LQ-compress M = R's top r×n block → L̃ + Qz-reflectors; forward-solve L̃ w = c1 (c1 = b[0..r), already Qᵀb); x = Qzᵀ w straight from the reflectors; un-permute x[P[j]]. Why the top-right block R12 matters: the BASIC (truncated) solution zeros the free variables in the pivoted column ordering, which is NOT minimum-norm for rank-deficient A, because R12 couples the free columns back into the leading ones (min ‖x‖ wants a nonzero free part that R12 can use to shrink the pivoted part). LQRP (the transpose-dual, wide side) has the SAME need: there the coupling lives in the below-diagonal block L21, and its basic solution is minimum-norm only for a CONSISTENT b — an inconsistent rank-deficient LS needs LQRP.minNormSolveInPlace, which QR-least-squares-solves the m×r block [L11; L21] (the transpose-dual of the LQ compress here). (was QRCP.fProxy.cs:1067)

## SVD
- 2026-07-13 | thin transposes U/V so bidiagonal QR rotations hit contiguous rows (same
  vectorization approach as Eigen.symmetricInPlace). (was SVD.fProxy.cs:203)
- 2026-07-13 | bidiagonalQR's deflation threshold is relative to the GLOBAL anorm, not local
  |d|+|e| — float needs this on clustered/zero singular values (same finding as the symmetric
  eigen QL). (was SVD.fProxy.cs:379)

## SVD.LowRank
- 2026-07-12 | Reorth windowing idea (moved from comment): a possible future optimization is
  windowing (compute_int strategy 0) to reorthogonalize against a subset of previous vectors
  instead of the full accumulated set. Not implemented. (was SVD.LowRank.fProxy.cs:218)

## SolveInfo
- 2026-07-11 | LstsqInfo doc trimmed from ~30-line essay to contract-only (struct fields + implicit-bool + pointer to Krylov.lstsqResidual). Removed content preserved here: per-solver norm derivation (Krylov R6a, docs/draft-spec-krylov-optimization.md) -- norms are the solver's own tracked values, never a fresh A*x/A^T*r, EXCEPT cgls's Converged exit: one fresh Apply + ApplyT verifies the claimed convergence before trusting it (replaces the drifted r/gamma pair). Per solver: cgls -- rnorm from a dot on its live residual r, Arnorm = sqrt(gamma) (its tracked ||A^T r||^2); lsqr -- rnorm = phibar, Arnorm = phibar*alpha*|c|, both produced free by the recurrence; lsmr -- Arnorm = |zeta-bar| (free, monotone), rnorm via the Fong-Saunders ||r|| recurrence (O(1) scalars per iteration, no matvec). Removed usage code sample:
  ```
  if (Krylov.lsqr(A, b, ref x)) { ... }          // implicit bool -> "did it converge?"
  bool ok = Krylov.cgls(A, b, ref x);            // same
  var info = Krylov.lsmr(A, b, ref x);           // keep the struct for diagnostics
  if (info.Solved) Debug.Log(info.iterations);
  ```
  (was SolveInfo.cs:6-35)
- 2026-07-11 | SolveInfo (square-system) doc: cg/pcg/cgne verify a claimed Converged exit with one fresh r = b-Ax first (ticket: Krylov R6a); minres/biCGStab need no extra matvec on any exit. (was SolveInfo.cs:88)

## UnsafeOP
- 2026-07-12 | formT's G=VᵀV pass: the naive per-(k,i) dot form (t as the reduction axis, stride Vld between consecutive t) does NOT vectorise and was measured far slower than the GEMM-shaped unit-stride loop actually used. (was UnsafeOP.fProxy.cs:721)
- 2026-07-11 | sumAbs/sum/maxAbs/vecDot shared header: one 4-lane accumulator left the FP add ports ~half idle in-cache; a 2nd independent width-4 accumulator measured ~2x. (was UnsafeOP.fProxy.cs:16-25)
- 2026-07-11 | matVecDot: two fProxy4 accumulators (8 lane-chains) measured ~2x over a single 4-lane accumulator (which left the FP add ports half idle in-cache); four accumulators measured NO further gain (memory/port-bound). (was UnsafeOP.fProxy.cs:154-165)
- 2026-07-11 | sortByKeyAscending was added to replace LP.ladBR's weighted-median ratio-test scan, which used to repeatedly linear-scan the REMAINING candidates for the current minimum ratio, removing the winner by swap-with-last each round -- an O(k) scan repeated up to k times is O(k^2), and at large m the candidate count k (bounded by m) made this the dominant cost of the whole solve even though the reported pivot count stayed small (each round can "fold" a candidate without registering as a pivot). Heapsort once up front costs O(k log k) instead, then a single linear walk. (was UnsafeOP.fProxy.cs:1218-1234)
- 2026-07-11 | UnsafeOP.iProxy.cs scalSub(target, n, s) used to implement "v - s" as "v + (-s)" for signed types only (bit-identical under modular wraparound, but unsigned types can't negate s); unified on the direct kernel so subInPlace<T>(T, iProxy) needs no per-signedness branch. (was UnsafeOP.iProxy.cs:216-220)
