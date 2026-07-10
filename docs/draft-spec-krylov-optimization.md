# Draft spec: Krylov solver optimization + preconditioner roadmap

Status: **APPROVED 2026-07-10** (user ruled on all seven open questions — see the RESOLVED
section below). Execution gate: benchmarks + committed state first (user ruling on Q3) —
starts after the LPBasis persistence feature lands. Nothing here is implemented yet. Covers (a) performance optimization of the existing Krylov solvers
(`Krylov.fProxy.cs`: cg/pcg, minres, biCGStab, cgls, lsqr, lsmr, cgne + the *Jacobi
convenience wrappers) and (b) a preconditioner roadmap beyond the shipped `fProxyBlockJacobi`.
LP-interior-point-specific preconditioning is deliberately NOT re-litigated here — that ground
is owned by `docs/research-lp-preconditioners.md`; this spec covers the general Krylov/BSR
surface and cross-references where they meet.

Claims are labeled **[verified]** (checked against code, git history, benchmark files, or a
fetched source) or **[judgment]** (literature-anchored estimate; must be A/B-measured before
being believed).

---

## RESOLVED QUESTIONS (user rulings 2026-07-10)

1. **Convergence-verification matvec: APPROVED.** +1 Apply at claimed convergence is an
   accepted amendment to the "never a fresh A*x" diagnostics contract. R6(a) is a go.
2. **Operator-interface evolution: (a) — break the interface pre-1.0.** `ApplyDot` goes onto
   `IfProxyLinearOperator` directly; the API is not final. User note: wants to review how the
   operator/kernel plumbing works — the implementing round's report must include a plain
   walkthrough of the interface change and what implementors must add.
3. **Preconditioner priority: benchmarks + committed state FIRST, then either track.** No
   preconditioner work on top of uncommitted state, and every round is A/B-benchmarked. The
   SSOR/IC(0) track proceeds in this spec's R1→R4 order once the gate is met.
4. **Symmetric-storage scope: v1 preconditioners are FULL-storage BSR only** (advice given to
   user, accepted direction): no new dedicated symmetric format. Rationale: the sweeps need
   row-ordered access to BOTH triangles — upper-only storage turns the lower sweep into a
   column-order scatter, and a purpose-built symmetric-sweep format would mean a third kernel
   family for one consumer. Symmetric-storage BSR (which exists for spMV) keeps its spMV
   kernels; a symmetric-storage matrix handed to a preconditioner SETUP pays a one-time
   mirror-to-full copy of the pattern+values (O(nnzb·b²), amortized over the whole solve
   lifetime, same lifecycle as the factorization itself). MKL/Eigen practice matches.
5. **Bit-exactness: NOT required for fused kernels.** Requirement is determinism (same result
   on every machine/run — deterministic-by-construction, FloatMode conventions unchanged), not
   bit-identity with the pre-fusion code. Rounding-only changes still declared per-commit.
   New fused kernels welcome. Kernel placement (user delegated): sparse kernels stay in the
   existing `UnsafeOP.Sparse.fProxy.cs` home; fused dense vector primitives go in `UnsafeOP`
   proper next to the axpy family. No new `UnsafeSparseOP` class — the file split already
   provides the separation.
6. **Chebyshev bounds (R7, still optional/demand-driven):** user concern "isn't solving for
   eigenvalues costlier than the solve itself?" — answer recorded: Chebyshev needs only a
   1–2-digit λmax estimate (~10–20 power-iteration matvecs, trivially cheap vs a full solve;
   λmin can be taken as λmax/κ_guess with graceful degradation), NOT an eigensolve. Decision
   deferred with R7 itself; if built, default = built-in cheap power estimate, caller override
   exposed.
7. **Benchmark budget: do NOT extend — make it faster.** New stencil + preconditioner-axis
   sections must displace redundant existing runs (the float==double duplicate cells that
   already told us what we need), keeping `LargeSparseBenchmark` at or under its current
   runtime. Treat total benchmark wall time as a budget to REDUCE while adding coverage.

---

## 0. Current state (surveyed 2026-07-09)

### Solver surface

`Krylov.fProxy.cs` (2117 lines): generic cores over `TOp : IfProxyLinearOperator` —
`cg<TOp>`, `pcg<TOp,TPre>`, `minres<TOp>`, `biCGStab<TOp>`, `cgls<TOp>` (damped),
`lsqr<TOp>`/`lsmr<TOp>` (damped, tracked-norm identities), `cgne<TOp>` (Craig/min-norm) —
each with the deliberate overload ladder (generic + dense + BSR + BSR-with-precomputed-Aᵀ +
damp/default forwarders). **The ladder is a locked decision — nothing in this spec adds or
removes rungs on existing solvers.** `pcg` already has the preconditioner slot
(`TPre : IfProxyPreconditioner`) and a shipped `fProxyBlockJacobi`; `lobpcg<TOp,TPre>` shares
the same `TPre` shape, so every new `IfProxyPreconditioner` serves both for free.

### Kernel status — what the SIMD reduction campaign did and did NOT cover [verified]

The 2026-07-08 campaign (`3668838`, `7af34d2`, `ba0d594`, `20d924a`) hand-SIMD'd the
**reductions**: `vecDot`/`vecDotRange`/`sum`/`sumAbs`/`maxAbs`, dense GEMV `matVecDot`, trsv,
symmetric tridiagonal matvec — the 2× `fProxy4` accumulator pattern (8 lane-chains). Dense CG
got 3.3× from it. Explicitly NOT covered:

- **All BSR spMV kernels** (`bsrMatVecB1..B6`, `bsrMatVecSymB*`, `bsrMatVecTB*` in
  `UnsafeOP.Sparse.fProxy.cs`): single accumulator per output row, no lane-chain splitting,
  relying on Burst auto-vectorization of the unrolled block body. No campaign SHA touched
  `Sparse/`.
- **`fProxyBlockJacobi.Apply`**: general runtime-`BR` double loop, single accumulator — no
  1/2/3/4/6 specialization, despite running every PCG iteration.
- **`fProxyBSROperator.ApplyBlock`**: no BSR block-multivector (SpMM) kernel at all — it loops
  per row through scalar `Apply` with two `Allocator.Temp` vectors. LOBPCG on BSR pays this.
- **`rowSquaredWeighted` / `columnNormsSquared`** (the Jacobi-diagonal builders): general
  runtime `BR×BC` loops.
- The axpy-family map primitives (`axpy`, `aypx`, `scalDiv`…) were *deliberately* skipped —
  they are element-wise maps that Burst already auto-vectorizes (kernel bench: axpy float
  71.8 vs double 36.8 GFLOP/s ⇒ SIMD firing). No work needed there.
- **No fusion anywhere**: every dot/axpy/copy in every solver loop is a separate full pass
  over its vectors.

### The load-bearing diagnostic [verified]

`benchmark-largesparse.txt` (BR=4, 1.5% fill, 40 fixed iterations): BSR CG at N=10240 runs
float 14.42 ms vs double 14.94 ms. **float ≈ double in TIME means float is moving half the
bytes in the same time — the spMV path is NOT bandwidth-bound; it is dependency/gather-latency
bound.** (Pure bandwidth-bound would put float near 2× double.) That is exactly the regime
where multi-accumulator ILP and prefetch have headroom — the same signal that preceded the
dense campaign's wins.

Benchmark hygiene caveat [verified]: the same file shows float PCG-Jacobi (6.19 ms) and float
BiCGStab (3.94 ms) "faster" than float CG at the same fixed 40 iterations — impossible for
solvers doing strictly more work per iteration. With tol=0 the convergence test can't exit, so
these are **breakdown-guard early exits** (float `⟨r,z⟩`/ω underflow). Any before/after bench
for this spec must record iterations-executed and exit status, not just wall time.

### Per-iteration pass accounting (the fusion target)

Counting full n-length vector sweeps (R=read, W=write) per iteration, excluding the operator
apply itself:

| solver | ops today | sweeps today | fused schedule | sweeps fused |
|---|---|---|---|---|
| cg | dot(p,Ap); x+=αp; r−=αAp; dot(r,r); p=βp+r | 9R+3W | fold dot(p,Ap) into Apply; one pass {x+=αp, r−=αAp, acc‖r‖²}; xpay | 6R+3W |
| pcg | cg's + M.Apply + dot(r,z) | +3R+1W | fold dot(r,z) into M.Apply | +1R+1W |
| cgls | dot(q,q); x+=αp; r−=αq; dot(s,s); p=βp+s | 9R+3W | fold dot(q,q) into Apply, dot(s,s) into ApplyT (AT path only); one update pass | 6R+3W |
| lsqr | 2×(xpay; dot; div); x+=·w; w=v−·w | ~12R+5W | 2×(fused xpay+normSq; div); one pass {x,w} over old w | ~9R+4W |
| minres | **6 CopyFrom** + 5 axpy + 2 div + 2 dots | ~25R+13W | rotate r1/r2 and w1/w2 buffers (4 copies → 0); fuse v=r2·(1/β); fuse w=(v−ε·w1−δ·w2)/γ into 1 pass | roughly halves |
| biCGStab | 2-stage p update; 2 x-axpys; r updates + 3 dots | ~14R+5W | p=r+β(p−ωv) one pass; fused axpy+normSq on both r stages | ~10R+4W |

MINRES is the biggest relative beneficiary: four of its six per-iteration full-vector copies
are pure history-shifts (`r1←r2, r2←y, w1←w2, w2←w`) replaceable by **struct-local buffer
rotation** (swap the `fProxyN` handles, move no data), and its 4-pass `w` update collapses to
one pass.

How much this is worth depends on fill [judgment, traffic model]: at BR=4 / 1.5% fill the spMV
streams ~6.7 MB/matvec vs ~0.5 MB of vector sweeps → fusion saves only a few %. At b=1
stencil-like fill (5-point Laplacian, nnz≈5n) vector sweeps are ~50% of total traffic → fused
CG saves ~13–18% wall clock, more for MINRES/PCG. Fusion is a 5–20% class win, cheap and
universal — not a 1.5×.

### Preconditioner status

Shipped: `fProxyBlockJacobi` (exact dense inverses of diagonal blocks, built via LU +
unit-vector solves, stored as explicit `DInv`), `fProxyNormalJacobi` (scalar diagonal for the
LP normal operator, rebuilt each IPM iteration via O(nnz) `rowSquaredWeighted`), column
equilibration for LS (`fProxyColScaledOperator` + `*Jacobi` wrappers). **There is no sparse
triangular-solve infrastructure of any kind** — no BSR forward/back substitution, no
factorization storage — which is the prerequisite for SSOR and IC(0)/ILU(0).

Customers: BSR SPD systems (cloth/FEM 3×3, Poisson/Laplacian 1×1 — the BSM spec's design
targets), the sparse LP normal equations (`standardFormInterior`: pcg + `fProxyNormalJacobi`,
inner cap `min(2m+20, 500)`), LOBPCG (`TPre` slot; measured: block-Jacobi cuts its iterations
~30%), and the LS stack (cgls/lsqr/lsmr at 20480×10240 confirmed).

---

## 1. Ranked recommendations

Ordered by expected value-for-effort. Iteration-count reduction (preconditioners) beats
constant-factor kernel wins wherever both compete for the same budget.

### R1. Fused vector kernels + copy elimination in the solver loops — **small effort, universal**

New `Blas`/`UnsafeOP` primitives (all `[NoAlias]` raw-pointer kernels; reductions inside them
use the exact 2× `fProxy4` accumulator pattern `vecDot` already uses):

- `axpyNormSq(a, x, ref y) → ‖y‖²` — y += a·x, return dot(y,y).
- `xpayNormSq(a, x, ref y) → ‖y‖²` — y = a·y + x, return dot(y,y) (lsqr/lsmr bidiag halves).
- `updateXR(a, p, ref x, q, ref r) → ‖r‖²` — x += a·p; r −= a·q; return dot(r,r)
  (cg/pcg/cgls/cgne's twin update + convergence dot in one pass).
- `scaledCopy(a, x, ref y)` — y = a·x (replaces CopyFrom+divInPlace pairs).
- `combine3(ref w, v, a, w1, b, w2, s)` — w = s·(v + a·w1 + b·w2) (MINRES w-update, 4 passes→1).
- Buffer rotation in minres (r1/r2, w1/w2) and any other pure history-shift copies: swap the
  local `fProxyN` struct handles instead of `Data.CopyFrom`. Caller-provided buffers are only
  entry handles; rotating locals inside the loop is contract-clean (the aliasing guard already
  ran).

Rewire all eight solver cores onto these. No public API change, no overload-ladder change, no
operator-interface change.

- **Expected win**: 5–20% solver wall clock depending on fill and solver (largest: minres, pcg,
  and every b=1/stencil workload); ~2% at BR=4/1.5% fill [judgment, traffic model above].
- **Cost**: one coder round. ~6 small kernels + loop rewiring + tests.
- **Risk**: low. Where the fused kernel preserves the existing accumulation order+pattern the
  result is **bit-identical** (axpy element order unchanged; the appended reduction uses the
  same fProxy4 fold as the separate `Blas.dot` did); state per-kernel which are bit-identical
  vs rounding-only, per repo convention.

### R2. BSR spMV ILP fix: multi-accumulator block-row kernels — **the float==double smoking gun**

Apply the campaign's lesson to `bsrMatVecB*`/`bsrMatVecSymB*`/`bsrMatVecTB*`: break the
single per-row dependency chain (currently one running sum per output row across ALL blocks of
the row, e.g. B6 is a 6-term chain per row per block, serialized block to block) into 2
independent accumulators (even/odd stored blocks), summed once at row end. For b=1 this is
literally the CSR-row dot getting the `vecDot` treatment. Same pass over the same data —
rounding-only, not bit-identical.

- **Expected win**: 1.2–1.8× on spMV time [judgment — the float==double diagnostic proves the
  kernel is latency-bound, but gather latency bounds the ceiling; must A/B at 1.5% AND stencil
  fills]. spMV is 60–95% of every sparse solve, so this multiplies everything.
- **Cost**: one coder round (15 kernels touched mechanically, one pattern).
- **Risk**: low-moderate: rounding-only change to every sparse solve; validated by existing
  dense-reference oracles. Watch the 4-vs-8-accumulator regression lesson — try 2, measure,
  stop.
- Fold into the same round: **specialize `fProxyBlockJacobi.Apply`** for b∈{1,2,3,4,6}
  (mirror the spMV unroll structure; it is a dense b×b matvec per block-row and runs every
  PCG/LOBPCG iteration) [expected: PCG total 1.05–1.15×, judgment].

### R3. Block triangular sweeps + block-SSOR preconditioner — **the infrastructure round**

The missing prerequisite for every serious SPD preconditioner. Build once, use twice (SSOR now,
IC(0) in R4):

- `BSR.sweepLower` / `BSR.sweepUpper`: block forward/back substitution over full-storage BSR —
  rows in order (reverse for upper), off-diagonal blocks apply as b×b matvecs against
  already-solved segments, diagonal solved via the **existing** `fProxyBlockJacobi` explicit
  inverses (b×b matvec, no per-row factorization). Sequential by construction — fine
  single-threaded, exactly the case the BSM spec anticipated.
- `fProxySSOR : IfProxyPreconditioner`: z = M⁻¹r with M = (D/ω+L) (ωD)⁻¹·ω/(2−ω) (D/ω+Lᵀ)
  — one forward sweep, diagonal scale, one backward sweep. **Setup = block-Jacobi's setup**
  (the D-block inverses — already shipped); no factorization, no breakdown risk, works for any
  SPD BSR. ω=1 (symmetric Gauss-Seidel) default; ω exposed.
- v1 scope: full-storage BSR only (see open question 4).

- **Expected win**: 1.5–2.5× iteration-count vs block-Jacobi on Laplacian/FEM-type SPD systems
  [judgment, standard literature range]; apply cost ≈ one extra spMV-equivalent per iteration
  (each stored block touched once across the two sweeps), so net wall-clock ≈ 1.2–1.7× where
  the iteration cut is at the high end — and it is the stepping stone to R4/R6.
- **Cost**: one coder round (sweeps + preconditioner + tests). No API-shape novelty: new struct
  implements `IfProxyPreconditioner`; add the same three-rung BSR pcg convenience overloads the
  `fProxyBlockJacobi` precedent established. Ladder untouched.
- **Risk**: low (no factorization; oracle = dense triangular solve on expanded matrix).

### R4. Block-IC(0) — **the iteration-count prize for SPD customers**

Incomplete Cholesky on the BSR block pattern (zero fill): factor A ≈ L·Lᵀ keeping only blocks
present in A's lower/upper pattern; diagonal blocks via the library's own dense `CHO` (b≤6),
off-diagonal updates as b×b GEMMs. Apply = one `sweepLower` + one `sweepUpper` from R3.
Breakdown (non-positive pivot block — guaranteed possible for general SPD, more so in float):
**Manteuffel-shift retry** — refactor A + αdiag(A) with growing α until the factorization
completes [verified concept: Manteuffel 1980; standard practice].

- **Expected win**: 2–5× iteration-count vs block-Jacobi on the SPD gallery
  (Meijerink–van der Vorst 1977 lineage; the classic PCG pairing) at ~1 spMV-equivalent apply
  cost → net wall-clock 1.5–3× on stable-pattern repeated solves [judgment — must be measured
  on `fProxyRandomSparseSPD` + `fProxyLaplacian2D`; the shift α trades robustness against
  iteration quality, and float needs a more conservative default].
- **Cost**: the largest item — factorization kernel + factor storage (mirror of A's upper
  pattern, nnzb·b² values) + shift-retry loop + tests. One heavy coder round after R3.
- **Risk**: moderate: float breakdown handling is genuinely fiddly; setup cost O(nnzb·b³-ish)
  only amortizes over repeated/many-iteration solves (per-frame re-value + re-factor is fine,
  pattern changes are not — same lifecycle as the BSM builder story).
- **ILU(0) corollary (non-symmetric)**: the same pattern-preserving factorization with block LU
  diagonal pivots instead of CHO gives biCGStab its preconditioner (frictional-LCP / MNA
  customers). Same sweeps, same storage shape, ~same size. BUT biCGStab today has NO
  preconditioner slot at all (no `TPre` overload) — adding a preconditioned biCGStab core is a
  new generic method (not a ladder change to existing rungs, same precedent as pcg beside cg).
  Build only when a non-symmetric customer materializes; SPD IC(0) goes first.

### R5. BSR SpMM (`ApplyBlock`) kernel — **LOBPCG's missing kernel**

A real block-multivector kernel: stream the BSR matrix ONCE, apply to k row-vectors
simultaneously (the dense operator already does this via `dotRows`; BSR falls back to per-row
scalar Apply with Temp allocs). LOBPCG holds 3k+guard vectors and calls ApplyBlock every
iteration.

- **Expected win**: up to ~2× on LOBPCG's matvec phase (matrix streamed once instead of k
  times; x-gather amortized) [judgment]; also removes per-call Temp churn.
- **Cost**: one focused coder round (one kernel family + dispatch; reuse the unroll pattern).
- **Risk**: low (oracle: equals k separate Applies).

### R6. Residual replacement / verified convergence — **float robustness, not speed**

cg/cgls/cgne update r recursively; in float the recursive residual drifts from b−Ax and can
report **false convergence** (or hide true convergence — see the tol=0 breakdown artifacts in
the benchmark). Two graded fixes:

- (a) **Verify-at-exit**: when the tracked residual first claims convergence, recompute
  r = b−Ax fresh (one Apply), re-test, continue if it fails. +1 matvec per solve, only at the
  claimed exit. Needs open question 1 resolved (contract amendment).
- (b) Full van der Vorst–Ye style periodic replacement (recompute r when the accumulated
  update-norm estimate crosses a threshold) [verified concept: van der Vorst & Ye, SIAM J. Sci.
  Comput. 22(3), 2000] — more machinery, only if (a) proves insufficient on the hard float
  instances.

- **Expected win**: correctness margin for the float builds (the library's differentiator);
  zero speed change on healthy solves.
- **Cost**: small (a) / moderate (b). **Risk**: (a) near-zero.

### R7 (optional). Chebyshev polynomial preconditioner — matrix-free niche

`fProxyChebyshev<TOp> : IfProxyPreconditioner`: k fixed Chebyshev iterations on [λmin, λmax]
as M⁻¹. No factorization, composes with ANY operator including fully matrix-free ones
(`fProxyNormalOperator`) and drops into pcg/lobpcg unchanged. Honest assessment: per matvec of
work it does NOT beat CG's own optimal polynomial — the real gain is amortizing the per-
iteration vector-op/dot overhead over k matvecs (the same motive as R1, by other means), plus
giving LOBPCG a matrix-free preconditioner option. Worth building only after R1–R4 land and
only if a customer (LOBPCG without assembled matrix, or the LP path) asks for it. λ-bounds:
open question 6.

### R8 (spike only). Software prefetch in spMV gather

Burst exposes `Unity.Burst.Intrinsics.Common.Prefetch` behind
`UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC` [verified: Burst 1.8 manual]. The repo already
uses an experimental-define precedent (`UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS` atop
Krylov.fProxy.cs). Prefetching `x[colInd[k+d]·b]` a few blocks ahead attacks exactly the
gather latency R2's diagnostic exposed. Gains on modern OoO cores are unpredictable
(0–30%) [judgment] — time-boxed A/B spike after R2, keep only if it clearly wins on both
dtypes at two fills.

---

## 2. Explicitly rejected (with reasons — the repo records negative results)

- **s-step / communication-avoiding Krylov (CA-CG, matrix-powers kernels)** — the headline
  win (fewer global reductions/synchronizations) is a distributed/parallel concern; this
  library's solvers are single-threaded inside one job, where a dot costs one cache-warm pass,
  not a latency barrier. The residual single-thread benefit (cache-blocking the matrix-powers
  kernel) requires Newton/Chebyshev basis-change machinery to avoid catastrophic float
  instability (Carson–Demmel) — large complexity for a win R1 captures more simply. REJECT.
- **Pipelined CG (Ghysels–Vanroose) as an algorithm variant** — hides reduction latency behind
  the matvec via asynchronous reductions; meaningless single-threaded, and it *worsens*
  rounding behavior. Its useful residue — merging vector updates into fewer passes — is exactly
  R1 without changing the recurrence. REJECT the variant, keep the fusion.
- **GMRES(m)** — O(m·n) basis memory + restart tuning; BiCGSTAB already covers the non-symmetric
  slot at flat memory (BSM spec already deferred it once). REJECT (unchanged).
- **AMG** — setup cost/complexity only amortizes on huge topology-stable grids; wrong scope for
  this library (BSM spec agrees). REJECT (unchanged).
- **IC(k)/ILUT (fill-in / drop-tolerance factorizations)** — dynamic fill = allocation-heavy,
  pattern-unstable, hostile to the zero-alloc/job-safe contract. IC(0)/ILU(0) fixed-pattern
  only. REJECT.
- **Hand-`fProxy4` SIMD inside the BSR block kernels** — b=3 (the FEM workhorse) misaligns
  with 4-wide lanes; the campaign's own lesson is accumulator-splitting first, intrinsics only
  where the layout fits. Revisit only if R2's measurements leave obvious headroom. REJECT for
  now.
- **RCM/bandwidth reordering now** — permutation-invariant for CG convergence; its value
  (gather locality, IC factor quality) is real but secondary — revisit WITH R4 if factor
  quality disappoints (BSM spec §6 already parked it). DEFER.
- **Double-accumulation inside float dots** — would quietly turn the float build into a
  half-double solver; violates the templated-everything policy the revised-simplex spec
  re-affirmed after the same mistake. REJECT.
- **Trimming or extending the solver overload ladder for preconditioner variants** — the
  ladder shape is a locked pre-release decision. New preconditioners arrive as new
  `IfProxyPreconditioner` structs + the established three-rung BSR pcg convenience pattern,
  never as new rungs on cgls/lsqr/lsmr (LS preconditioning stays operator-composition:
  `fProxyColScaledOperator` precedent). CONSTRAINT, not a work item.

---

## 3. Staged implementation plan (coder-agent rounds)

Each round: templates → `Tools/run-tests.ps1` (auto-regen) green → A/B bench → commit. All
rounds independent of the LP research doc's track (which can interleave).

**Round 1 — R1 fused kernels + copy elimination.**
Scope: `UnsafeOP` fused primitives, `Blas` wrappers, rewire the 8 solver cores, minres/lsqr
buffer rotation. Oracles: full suite (`*Krylov*`, `*Sparse*` filters) — solutions must match
current to solver tolerance; add explicit **bit-identity tests** for the fusions that preserve
accumulation order (assert exact equality of x and rnorm vs a pinned pre-fusion reference
path on a fixed seed), tolerance tests for the rest. Bench: `LargeSparseBenchmark` before/after
+ new stencil section (`fProxyLaplacian2D`, b=1) where the win should be visible; record
iterations+status per timed sample (fixes the tol=0 artifact while in there).

**Round 2 — R2 spMV multi-accumulator + block-Jacobi Apply specialization.**
Scope: 15 BSR kernels + `fProxyBlockJacobi.Apply` unrolls. Oracles: existing dense-reference
spMV tests (tolerance, not bit); SPD gallery solves converge to same solutions. Bench: spMV×50
section, float AND double, both fills — expect the float/double gap to OPEN (the diagnostic
inverts: if float pulls ahead of double, the latency bound moved toward bandwidth).

**Round 3 — R3 triangular sweeps + block-SSOR.**
Scope: `BSR.sweepLower/sweepUpper`, `fProxySSOR`, three pcg convenience overloads, docs.
Oracles: sweep vs dense triangular solve on expanded random BSR (both dtypes, b∈{1,2,3,4,6});
SSOR-PCG on `fProxyRandomSparseSPD` + `fProxyLaplacian2D`: converges to cg's solution,
iteration count ≤ block-Jacobi's on the same instance (assert with margin), M-SPD sanity
(⟨r,z⟩>0 throughout). Bench: PCG section grows a preconditioner axis (none/Jacobi/SSOR),
metric = iterations AND wall clock.

**Round 4 — R4 block-IC(0).**
Scope: factorization (upper-pattern factor storage, CHO diagonal blocks, GEMM block updates),
Manteuffel-shift retry, `fProxyIC0 : IfProxyPreconditioner`, overloads, tests. Oracles: on
small instances expand L·Lᵀ dense and assert it matches A ON THE PATTERN (the IC(0) defining
property); PCG-IC0 iteration count < SSOR < Jacobi on the Laplacian gallery; float shift-retry
exercised by a deliberately hard instance (near-singular SPD via `fProxyRandomSparseSPD`'s
conditioning knobs). Bench: same axis as Round 3.

**Round 5 (parallelizable with 3/4) — R5 BSR SpMM + R6(a) verify-at-exit** (pending open
question 1). Oracles: SpMM equals k scalar Applies (bit or tolerance per implementation
choice); LOBPCG iteration counts unchanged, wall clock down; verify-at-exit: construct a float
instance where the recursive residual lies (long ill-conditioned solve), assert the guarded
exit reports honestly.

**Round 6 (optional, demand-driven) — R7 Chebyshev / R8 prefetch spike.** Time-boxed; keep only
on a clear two-dtype, two-fill win.

Total bench budget: keep `LargeSparseBenchmark` ≤10 min by making the new stencil +
preconditioner-axis sections replace the current redundant double-runs where float==double has
already told us what we need (open question 7).

---

## 4. References

- Eisenstat, "Efficient implementation of a class of preconditioned conjugate gradient
  methods", SIAM J. Sci. Stat. Comput. 2(1), 1981 — the SSOR trick (R3 follow-up: makes
  SSOR-PCG's per-iteration cost ≈ unpreconditioned CG's; adopt only if R3's plain form
  measures well). [verified via epubs.siam.org/doi/10.1137/0902001]
- Meijerink & van der Vorst, "An iterative solution method for linear systems of which the
  coefficient matrix is a symmetric M-matrix", Math. Comp. 31, 1977 — IC(0)+CG. [literature]
- Manteuffel, "An incomplete factorization technique for positive definite linear systems",
  Math. Comp. 34, 1980 — shifted IC. [literature]
- van der Vorst & Ye, "Residual replacement strategies for Krylov subspace iterative methods
  for the convergence of true residuals", SIAM J. Sci. Comput. 22(3), 2000. [literature]
- Chronopoulos & Gear 1989 (s-step CG); Ghysels & Vanroose 2014 (pipelined CG); Carson &
  Demmel 2014 (CA-Krylov stability) — the rejected family. [literature]
- Unity Burst manual, "Burst Intrinsics Common class" — `Common.Prefetch` behind
  `UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC`. [verified:
  docs.unity3d.com/Packages/com.unity.burst@1.8/manual/csharp-burst-intrinsics-common.html]
- Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., ch. 10 (preconditioning),
  ch. 12 (polynomial preconditioners). [literature]
- Internal: `docs/research-lp-preconditioners.md` (LP-IPM preconditioning track — owns §1–§8
  of that problem); `docs/dev/spec-sparse-bsm.md` (BSR design + deferred-preconditioner tier);
  `docs/dev/perf-vectorization-lessons.md` (the float==double diagnostic, accumulator
  sweet-spot, axpy-vs-dot); memory `iterative-solver-overload-ladder` (locked ladder);
  `TestResults/benchmark-largesparse.txt`, `benchmark-kernels.txt` (untracked measurements
  cited above).
