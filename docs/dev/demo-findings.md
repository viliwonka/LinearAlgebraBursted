# Demo stress-test findings (2026-07-11 overnight)

Library used as an actual consumer would: realtime Unity demos under `Assets/Demos/`, float
only, all math inside `[BurstCompile]` jobs within a frame budget, MonoBehaviour ↔ job
interop. Per-demo notes on what was awkward, wrong, or annoying. Written while coding, from
docs first (README → docs/features/*) the way a new user would.

> **Status 2026-07-12 (owner-approved fixes shipped):** #3b (NativeArray bridge: view ctors +
> CopyTo/CopyFrom(NativeArray)), #3c (IsCreated), and the matrix zeroInPlace/fillInPlace gap
> (Demo 5) are FIXED — all dtypes, tested (BridgeFillTests/BoolBridgeTests), documented in
> dense-types.md/comp-elementwise.md/CHANGELOG. The truss and circuit demos now consume
> IsCreated and a solve-through-view respectively. Still open: warm-struct RunByRef doc/API
> story (ties into coherence P.1), pcg convenience-overload job trap, control.md fProxyLQRState
> leak, LOBPCG penalty-pinning doc warning, LP.lad tau hybrid overload.

## Cross-cutting (found before any demo compiled)

1. **`docs/features/control.md` leaks template names into a user-facing doc**: lines 10-13
   say `fProxyLQRState` and "generated per-dtype" language; every other feature doc speaks
   float (`floatLPCache`, `IfloatLinearOperator`). A user copying `new fProxyLQRState(n,
   allocator)` gets a compile error. Should read `floatLQRState`/`doubleLQRState`.
2. **`subInPlace(this T place, T fromB)` parameter name is ambiguous** (OP.Component:
   ~line 61): `fromB` reads as "subtract place FROM b". `addInPlace` at least has a
   `// place += from.` comment; `subInPlace` has none. Rename param to `subtrahend`/`b` +
   one contract comment (`place -= b`) would remove the doubt. Same family: `scaleAddInPlace`
   vs `addScaledInPlace` differ only by word order for two different operations — correct
   but takes a double-take every time; the `// y += a*x` / `// y = a*y + x` comments carry
   all the meaning.
3. **Job-safe copying exists but is invisible.** Wanted a pristine copy of the design
   matrix for residual recomputation after a destructive `QR.solveInPlace` (which eats both
   A and b) with no arena in the job. The right tool exists — the standalone copy ctor
   `new floatMxN(in orig, Allocator.Temp)` (floatMxN.cs:93) — but no doc mentions it:
   dense-types.md documents only arena-tracked `Copy()`/`TempCopy()` ("only valid on
   arena-tracked instances"), so a reader concludes job-side copying isn't offered and
   hand-loops (I did, until grepping the struct source). Also asymmetric: `floatN.CopyTo(in
   floatN)` (copy into existing buffer) has no `floatMxN` counterpart.
3b. **No zero-copy view over `NativeArray<float>`.** Game state naturally lives in
   NativeArrays (job I/O); every trip into the library means element-copying into a
   `floatN`/`floatMxN` (both own their UnsafeList). A read-only wrap/view (or a
   `CopyFrom(NativeArray<T>)` at least) would remove the per-frame boundary copy every
   demo here hand-rolls. In the control demos I skipped `floatN` for the state vector
   entirely and unrolled K·x by hand — 6 states is fine, but it shows the boundary
   friction: the natural "small hot state in NativeArray, math in library types" split
   has no supported bridge.
3c. **No `IsCreated` on floatN/floatMxN.** Every Unity native container has it; the demo
   code reflexively wrote `lambda.IsCreated` and got CS1061 (the only compile errors in
   ~1400 lines of first-pass demo code). The structs do track validity internally
   (`Data.IsCreated`), it's just not surfaced. One-property template addition would let
   MonoBehaviour lifecycle code treat library types like every other native container.
4. **README quick start is arena-only; the realtime/job workflow is the second-class
   citizen.** The pattern every one of these demos needs — `new floatMxN(m, n,
   Allocator.Temp)` inside a Burst job, in-place ops only, no arena — is real and works
   (tests use it everywhere) but a user learns it from reading tests, not docs. lp-lad.md
   is the exception (its job-safety paragraph is exactly what's needed). A short "inside a
   job" section in README or dense-types.md would save every game user an hour.

## Demo 1 — least squares L2 surface fit (`01_LeastSquares`)

Setup: m∈[16,4096] noisy animated points, plane (n=3) or quadric (n=6) design matrix,
`QR.solveInPlace` per frame in one Burst job; gizmo surface + on-screen readout.

- **Good**: the solve itself is a three-liner; `DirectSolveInfo`'s implicit bool is nice
  (`Stats[1] = info ? 1f : 0f` just works). `floatGaussian` sampler struct is job-friendly
  and documented in random.md.
- Destructive-solve ergonomics: fine once you know it, but the "recompute residual
  yourself from the model" dance is boilerplate every fitting demo repeats (see
  cross-cutting #3). `LstsqInfo` (Krylov side) carries `rnorm`; the direct solvers'
  `DirectSolveInfo` carries nothing comparable — understood why (diag fields only from
  already-computed numbers), but the asymmetry is felt when you switch routes.

## Verification

Every demo job has a headless EditMode smoke test (`Assets/Demos/Tests/DemoSmokeTests.cs`,
9 tests) that runs it with real inputs over a short simulated horizon and asserts solver
success AND a physical outcome (pole stabilizes, drone reaches target, cloth sags, L1
intercept beats L2 under biased outliers, source voltage enforced...). All 9 green in ~1s.
Two test iterations were needed, both instructive: the cloth-sag threshold assumed cm-scale
sag from mm-stiff springs (test bug), and the truss test exposed the float penalty-pinning
spectrum collapse documented in Demo 4's findings (demo design bug, library doc gap).
Interactive behavior (sliders, gizmos) still needs an in-editor play session.

## Demo 2 — LAD L1 vs L2 robustness (`02_LeastAbsoluteDeviation`)

Setup: animated plane + upward-biased outliers; per frame fit L1 (`LP.lad` /
`ladBR`/`ladFN` with τ slider) and L2 (`QR.solveInPlace`) on the same design matrix, draw
both planes.

- **Good**: `LP.lad(in A, in b, ref x, out obj)` is a one-liner and takes `in A` — being
  non-destructive while QR's route destroys A/b means ORDER MATTERS when you run both fits
  on one matrix: L1 first, then let QR eat the buffers. Works, but nothing warns you if you
  do it backwards — you'd silently fit L1 on QR's scratch garbage. A destroyed matrix
  carries no "I'm scratch now" flag (fair — zero-cost struct — but this is the sharpest
  edge the destructive convention has when composing two solvers).
- `tau` is `double` in the float API (`ladBR(in floatMxN, in floatN, double tau, ...)`),
  and `objective` is `out double` everywhere. Defensible (accumulation), but a float-only
  caller sprinkles `(float)` casts on every readout.
- The hybrid `LP.lad` has no τ overload — median only. Want τ ≠ 0.5 → you pick the engine
  yourself (`ladBR` vs `ladFN`) and re-implement the m-based routing the hybrid already
  knows. A `LP.lad(in A, in b, double tau, ...)` forwarding to the same crossover would
  cost nothing.

## Demo 3 — realtime LP, village economy (`03_LinearProgram`)

Setup: 4 products / 3 shared resources + demand caps (7×4 LP), capacities and profits on
sliders, warm-started re-solve every frame (`LP.solve(..., ref LPBasis, ref floatLPCache)`)
in a Burst job.

- **THE interop finding of the night: every warm-state struct (LPBasis, floatLPCache,
  floatLQRState) carries load-bearing SCALAR fields (`populated`, `builtVersion`,
  `factorsValid`...) that the solver mutates through `ref`.** A Unity job struct is copied
  by value, so with plain `job.Run()` those scalar mutations die with the job copy — the
  NativeArray contents persist but the flags say "cold" — and every frame silently pays a
  full cold solve while the API happily reports Optimal. The working pattern is
  `IJobExtensions.RunByRef(ref job)` + copying the structs back out of the job afterwards.
  NOTHING in the docs or XML comments says this; lp-lad.md's own warm-loop example is
  main-thread-only and doesn't mention jobs. This deserves a doc section ("warm state
  across jobs") or, better, an API shape that keeps mutable state behind a pointer so
  by-value copies stay coherent.
- `cache.matrixVersion` must be bumped when **c** changes, not just A/senses — lp-lad.md
  says "A's coefficients/senses or c change" — correct, but easy to miss that the
  *objective* is part of the cached computational form; my first mental model was
  "matrix version = A only". The field name says matrix; the contract says matrix+cost.
  `formVersion` or `problemVersion` would name the contract.
- P.1 (coherence audit) confirmed from the consumer seat: LPBasis/floatLPCache/floatLQRState
  are the only things in these demos needing manual Dispose bookkeeping; everything else
  demo-side lives in NativeArrays or fresh Temp. Arena factories for the warm structs would
  kill the last leak footgun.
- Small: `new floatLPCache(n, m, ...)` takes (n, m) while every matrix in sight is m×n —
  I had to read the ctor doc to be sure. `IsValid` throwing on mismatch saves the day at
  runtime.

## Demo 5 — cart-pole LQR (`05_PendulumLQR`)

Setup: per frame — re-linearize cart-pole about upright from slider params, warm
`Control.lqr(..., ref K, ref floatLQRState)`, then 4 RK4 substeps of the full nonlinear
dynamics under u = −K·x (clamped). Kick/reset buttons.

- **`docs/features/control.md`'s `fProxyLQRState` is confirmed wrong for users** — the
  generated type is `floatLQRState` (cross-cutting #1). Copy-pasting the doc line does not
  compile.
- **No public matrix zero/fill**: `floatComp.zeroInPlace` exists for `floatN` only
  (stranded in UtilityOP.cs — not even in the Comp family file), nothing for `floatMxN`,
  no `fillInPlace(mat, s)` either. Reusing persistent A/B/Q/R across frames therefore
  means hand-zeroing stale entries; idiomatic escape is "allocate fresh Temp every call —
  construction zero-initializes". That idiom is fine (and what the library's own tests
  do) but it is documented nowhere, and `mulInPlace(A, 0f)` — the "obvious" workaround —
  is a NaN propagator. Add `zeroInPlace`/`fillInPlace` over IUnsafefloatArray (matrix +
  vector, all dtypes) — it's one template method.
- Warm LQR across jobs: same RunByRef story as Demo 3 (`floatLQRState.populated` is a
  scalar). Once known, the pattern is mechanical.
- **Good**: `Control.lqr` itself is exactly the right shape for games — `in` matrices,
  `ref K`, warm overload never allocates, `LQRInfo` implicit bool. The warm path visibly
  converges in 1-3 Riccati iterations on slider drags (vs ~15-25 SDA cold). The
  "terminal S written back only on Converged" contract meant a mid-drag divergence never
  poisoned the next frame — noticed, appreciated.

## Demo 5b — DOUBLE pendulum on cart (`05_PendulumLQR/DoubleCartPoleLQRDemo`)

Setup: 6-state double inverted pendulum; the upright linearization is built by solving
M0·W = [G|F] with the **multi-RHS** `CHO.solveInPlace(ref A_to_L, ref B_to_X)` (3×4 RHS),
and every RK4 derivative evaluation solves the 3×3 configuration-dependent mass matrix
with the vector `CHO.solveInPlace`. 32 small Cholesky solves per frame + warm Riccati.

- This is the "library inside the integrator" pattern and it reads beautifully: build M,
  build rhs, `CHO.solveInPlace(ref M, ref rhs)`, rhs is now q̈. The destructive
  convention is at its best here — no copies wanted, everything is per-stage Temp.
- Multi-RHS overload discovered only by grepping CHO.float.cs; solvers.md — worth
  checking it mentions matrix-RHS forms (didn't verify).
- `DirectSolveInfo` returned per stage is discarded except the linearization one; a
  cheap `.Solved` check per stage would cost branches — fine as-is.

## Demo 6 — planar drone LQR (`06_DroneLQR`)

Setup: 6-state planar quadrotor, 2 inputs (thrust delta, torque) mapped to two clamped
rotor forces; warm LQR about hover per frame; RK4 nonlinear sim tracking an orbiting
target; wind gusts.

- Mostly re-confirms Demo 5's patterns at n=6, m=2 — nothing new broke; the multi-input
  gain (2×6 K) worked first try, `R` being 2×2 diagonal via indexer is fine.
- K·(x−x_ref) is hand-unrolled again (see cross-cutting #3b): with no NativeArray↔floatN
  bridge, building floatN temporaries just to call `Blas.dot(K, e, ref u)` for a 2×6
  gain costs more ceremony than the two dot products it replaces. For n≥20 state
  (e.g. MPC later) the bridge will stop being optional.
- Would have liked `Control.lqr` to accept `Q`/`R` as vectors-of-diagonals overload —
  every game tuning session uses diagonal weights; building the full n×n each frame just
  to carry 6 numbers is noise (and invites the stale-entry bug the fresh-Temp idiom
  works around).

## Demo 7 — implicit mass-spring cloth (`07_SpringSystem`)

Setup: 12×10 grid (360 dof), A = M + h²kL assembled once into symmetric lower-block BSR
(3×3 blocks), per frame: nonlinear spring forces → rhs, `Krylov.pcg` + `floatIC0` with
caller scratch, integrate. Pinning via 1e7 penalty masses.

- **`AddValue`-based assembly is genuinely pleasant** — scalar global indices, duplicate
  summing, no pre-merging: the whole Laplacian assembly is ~15 lines. The (now-lower)
  one-triangle authoring rule for `ToBSRSymmetric` is easy to satisfy when edges are
  normalized (a<b → write at (b,a)).
- **The pcg convenience overloads are a job trap**: `pcg(in A, in M, in b, ref x, maxIter,
  tol)` internally does `b.floatTempVec(...)` — arena-tracked temp allocation off the
  RHS vector. With a standalone `Allocator.Temp` b inside a job (no arena anywhere), that
  path can't work; the caller must know to use the explicit-scratch overload (4 extra
  `ref floatN` params for pcg, SEVEN for pbiCGStab). Nothing in the signature or docs
  distinguishes "arena-tracked-b only" from "job-safe" — the overload list looks
  interchangeable. Suggest: doc flag on the convenience overloads, or make them detect a
  non-tracked b and throw a clear message. (Same pattern presumably across cg/minres/
  cgls/... — checked pcg and pbiCGStab only.)
- Scratch-vector arity: 4 vectors for pcg, 7 for pbiCGStab, each a separate `ref` param.
  A per-solver scratch struct (`floatPCGScratch(n, allocator)`) would collapse the
  signatures and stop callers mis-ordering same-typed args (nothing catches swapping
  `ref p` and `ref Ap`).
- Matrix REBUILD on stiffness change means arena dispose + full reassembly + new IC0 —
  fine at this size, but there's no "update values, keep pattern" path on floatBSR
  (values are exposed; a `Values`-rewrite idiom probably works — undocumented whether
  IC0 caches or references A's values). For per-frame-varying stiffness this matters.

## Demo 4 — truss stability, sparse eigen (`04_TrussStability`)

Setup: 9-node house frame, 2×2-block symmetric-lower BSR stiffness matrix, toggleable
diagonal braces, per-frame warm `Eigen.lobpcg` (floatBSROperator + floatBlockJacobi,
k=4) with the cache carrying eigenvectors across frames; λ₁→0 = mechanism detector,
mode shapes animated.

- The generic `lobpcg<TOp,TPre>(in A, in M, ref cache, k, tol, maxIter)` + arena cache
  worked exactly as the structural-stability memory promised; warm restarts drop to a
  handful of iterations while dragging EA.
- `arena.floatLOBPCGCache(n, k)` internally sizes for 3k block vectors — fine, but
  the cache exposes `X`/`lambda` plus ~10 other public scratch matrices with terse names
  (`Xnext`, `APnext`...). Which fields are "results" vs "internals" is only knowable
  from the doc comment; a result-view property pair (`eigenvalues`/`eigenvectors`) would
  make the cache self-explanatory.
- Assembling 2×2 bar-element blocks via 9 `AddValue` calls per bar is fine; an
  `AddBlock(bi, bj, in float2x2)`-style overload taking Unity.Mathematics fixed types
  would be the natural game-facing sugar (AddBlock exists but takes `float*` or a full
  `floatMxN` — neither is convenient for a hand-computed 2×2).
- **Penalty pinning ate the spectrum in float — the best runtime finding of the night.**
  With the standard "big number on pinned diagonals" trick at 1e6, `Eigen.lobpcg`
  CONVERGED and returned λ₁ = exactly 0 on a provably stiff braced truss — on BOTH
  symmetric and full storage (so not a storage bug; sym/full spMV agreed to float
  precision). Cause: Rayleigh-Ritz Gram entries scale like penalty² ≈ 1e12; float eps at
  that scale is ~1e5, so the true O(EA)≈O(5) eigenvalues sit eleven orders below the
  Gram noise floor and get annihilated — and nothing in `LOBPCGInfo` distinguishes this
  from a genuine zero mode (status says Converged, and a zero eigenvalue is exactly what
  the demo's "mechanism detected!" path looks for → silent FALSE POSITIVE). Fix: keep
  penalties within ~3 decades of the physical stiffness (1e3 works). Suggest an
  eigen.md/lobpcg doc warning: "float + penalty-style Dirichlet pinning: penalty² must
  stay well inside float range or small eigenvalues collapse to 0" — every FEM-style
  user will hit this, and the failure mode is indistinguishable from the instability
  they're testing for. (A proper reduced-dof assembly avoids it entirely; a doc note on
  imposing Dirichlet dof in BSR assembly would cover both.)

## Demo 8 — RC-grid circuit, MNA (`08_Circuit`)

Verdict on "does it make sense": **yes** — it's the one demo that naturally produces a
symmetric INDEFINITE system (voltage-source rows with zero diagonal), i.e. the only
realtime consumer of `Krylov.pbiCGStab` + `floatILU0` in the set. RC diffusion heatmap
reads well visually.

- MNA's zero-diagonal constraint rows require explicitly storing the zero diagonal
  entries (ILU0's "every diagonal block stored" contract) and ordering constraint rows
  last so elimination fills their pivots — standard MNA lore, but worth one line in the
  ILU0 doc ("indefinite systems: store explicit zero diagonals; expect shift retries
  otherwise").
- 7 scratch `ref` vectors for job-safe pbiCGStab (see Demo 7 note) — at this arity the
  call site is genuinely hard to read.
- Everything else identical in feel to Demo 7 — builder assembly, arena preconditioner
  factory, per-frame Temp rhs. The BSR/Krylov surface is coherent across SPD and
  indefinite paths.
