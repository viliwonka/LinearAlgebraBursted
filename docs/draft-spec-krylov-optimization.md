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

### R4. ~~Block-IC(0)~~ — **DROPPED by user 2026-07-10**

User ruling: dropped from the plan ("drop R4"). Rationale from the discussion: the largest
and fiddliest item (factor storage, Manteuffel shift-retry, float breakdown handling), and
its natural customer — the implicit-Euler cloth / Poisson demo from the dynamics roadmap —
does not exist yet, so it would be tuned against synthetic matrices only. Do not revive
without a user ask; if revived, the trigger is a real per-frame SPD workload (cloth demo)
underperforming with SSOR. The ILU(0) corollary and the RCM revisit trigger (both tied to
R4) are dropped with it. Original design kept below for that eventuality.

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

**RESOLVED 2026-07-10 — see the R2/R8 addendum immediately below: pairing REVERTED (no
reproducible win), prefetch REJECTED (consistently slower).**

---

## 1b. R2/R8 addendum (2026-07-10): pairing measured and REVERTED, prefetch spiked and REJECTED

**Method** [verified, this round]: a SCRATCH microbenchmark (`_ScratchSpmvVariants.cs`, hand-written,
non-template, NOT part of the shipped suite, deleted after use — see disposition below) with local
copies of the `bsrMatVecB1`/`bsrMatVecB4` kernel bodies as four variants — paired (R2's shipped
form at the time), unpaired (pre-R2 original, reconstructed from `git show 02cdc5c:...`),
paired+prefetch, unpaired+prefetch (b=1 only has unpaired forms, since R2 had already reverted
pairing there) — run via `Bench.Time` (1 warmup + 4 timed, median) at reps=50 per sample, on the
SAME two matrices Round 2's own bench section used: `fProxyRandomSparseSPD` BR=4/1.5% fill and
`fProxyLaplacian2D(1,N)` b=1 stencil, both N=10240, both dtypes. **The prefetch intrinsic required
a project-wide `UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC` Scripting Define Symbol, NOT a
per-file `#define`** — a caller-file `#define` cannot reach into the separate `Unity.Burst`
assembly where `Common.Prefetch` is itself `#if`-gated, so the existing
`UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS` per-file-`#define` precedent atop several
`TemplateSource` files was discovered to be **inert/decorative for this purpose** (confirmed no
`Loop.ExpectVectorized`/`ExpectNotVectorized` call sites exist anywhere in the codebase either —
those defines have never actually gated anything). The define was toggled on/off project-wide via
a temporary Editor script for this spike only and fully reverted afterward (`ProjectSettings.asset`
verified byte-identical to its committed state via `git diff` after cleanup).

The whole variant set ran 3 TIMES (this machine has shown 15–25% swings run-to-run):

| dtype | variant | Run1 med(ms) | Run2 med(ms) | Run3 med(ms) |
|---|---|---|---|---|
| float | B1_Unpaired | 0.8346 | 0.8309 | 0.8983 |
| double | B1_Unpaired | 0.8643 | 0.9845 | 0.9724 |
| float | B1_Unpaired_Prefetch | 1.1476 | 1.3004 | 1.2673 |
| double | B1_Unpaired_Prefetch | 1.0349 | 1.1296 | 1.1273 |
| float | B4_Paired | 21.0006 | 22.9481 | 23.1283 |
| double | B4_Paired | 24.0095 | 25.4203 | 24.9590 |
| float | B4_Unpaired | 23.0065 | 23.2851 | 23.1358 |
| double | B4_Unpaired | 23.7988 | 25.1325 | 25.0678 |
| float | B4_Paired_Prefetch | 20.0101 | 25.5371 | 25.3032 |
| double | B4_Paired_Prefetch | 26.6901 | 27.0152 | 26.7606 |
| float | B4_Unpaired_Prefetch | 25.4151 | 25.5200 | 24.9880 |
| double | B4_Unpaired_Prefetch | 27.0207 | 26.9744 | 26.8806 |

Prefetch distance pre-check (time-boxed, B4_Paired_Prefetch/float only, single sample each, per
the spec's own "try 2–4, don't sweep exhaustively" instruction): dist=2 → 24.97ms, dist=3 →
25.19ms, dist=4 → 23.81ms. dist=4 used for all rows above.

**Verdict (i) — pairing for b=4 (and by proxy b=2/3/6): NEUTRAL, no reproducible win.** The
paired-vs-unpaired float delta shrinks run over run (−8.7%, −1.5%, −0.06%) — consistent with
Run 1 being a warm-up/noise outlier rather than a real effect — and double shows NO consistent
direction (paired slower in runs 1–2, faster in run 3, all within ±1.2%). Every paired-vs-unpaired
difference is smaller than the run-to-run swing measured on the SAME kernel across repeats (float
B4_Paired alone: 21.00/22.95/23.13ms, a ~10% spread with no code change at all) — exactly the
"BR=4 section too machine-noisy to attribute" caveat R2's own commit message flagged. **Action:
REVERTED** the 2-accumulator pairing in `bsrMatVecB{2,3,4,6}`, `bsrMatVecTB{2,3,4,6}`,
`bsrMatVecSymB{2,3,4,6}` back to the single left-to-right accumulator fold (bit-identical to the
general fallback again, matching b=1's own already-settled non-pairing) in
`UnsafeOP.Sparse.fProxy.cs`. The R5 SpMM kernels (`bsrMatMatB*`/`bsrMatMatSymB*`) documented
themselves as bit-identical-per-row to repeated scalar `Apply` calls "with the same pairing where
the scalar kernel pairs" — reverting the scalar kernels without also reverting these would have
broken that documented invariant, so `bsrMatMatB{2,3,4,6}`/`bsrMatMatSymB{2,3,4,6}` were reverted
in lockstep (unpaired, single accumulator, `rv`-loop preserved from R5). Everything else R2/R5
added is UNCHANGED: `blockJacobiApplyB*` unrolls, `ApplyDot`, the SpMM kernel family and dispatch
itself, R6a verify-at-exit. Determinism: rounding-only reversion (restores the ORIGINAL
bit-identical-to-general-fallback form R2 moved away from), same FloatMode conventions.

**Verdict (ii) — prefetch: NO, does not win on either dtype at either fill. REJECTED, not
shipped.** Consistently and substantially SLOWER — 11 of 12 dtype/fill/pairing comparison cells
across all 3 runs show a slowdown with prefetch on, the sole exception being the same noisy
Run-1 float-B4-Paired cell already flagged above as an outlier:
- b=1 stencil: +37–56% slower (float), +15–20% slower (double), every run, both dtypes — the
  worst cells. A b=1 row here has only ~3 stored blocks; the bounds-check-and-prefetch-call
  overhead is pure loss on a row that short.
- BR=4/1.5% fill: +6–13% slower on 11 of 12 (paired/unpaired × float/double × 3 runs) cells;
  the only faster cell (float B4_Paired_Prefetch, run 1, −4.7%) is the same run/cell already
  identified as a warm-up outlier for the non-prefetch comparison above.
No further distance sweep or kernel change pursued — the direction is unambiguous and time-boxing
per the spec applies. **Action: NOT added to any real kernel.** No `UNITY_BURST_EXPERIMENTAL_
PREFETCH_INTRINSIC` code exists anywhere in `TemplateSource` after this round; the temporary
project-wide define used for the spike was fully reverted (see Method above).

**Disposition**: `_ScratchSpmvVariants.cs`, the temporary `ScratchBurstDefineToggle` editor
helper, and the temporary project-wide Burst define are all gone — verified via `git status`
showing no trace after cleanup (same "like it never shipped" treatment as the removed
`pcgEisenstat`, §3b). Real-kernel diff: `UnsafeOP.Sparse.fProxy.cs` (template) — pairing removed
from 30 kernels (15 scalar `bsrMatVec{,T,Sym}B{2,3,4,6}` + 15 SpMM `bsrMatMat{,Sym}B{2,3,4,6}`
mirrors) — plus the regenerated `float`/`double` `Source/Generated` twins. Oracle: full suite,
unchanged (these kernels are covered by the existing dense-reference spMV tests and the R2/R5
bit-identity/oracle tests, which assert against the general fallback — a target this reversion
restores exact equality with, so no test file changes were needed).

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

**Round 4 — R4 block-IC(0). DROPPED by user 2026-07-10 (see R4 above); rounds 5+ renumber up.**
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

## 3b. R3b addendum: Eisenstat-SSOR PCG derivation (2026-07-10) — **REMOVED, negative result**

**VERDICT (2026-07-10, benchmark-falsified): `Krylov.pcgEisenstat` was implemented, tested (18
tests x float/double, all passing — the math and the transformed-variable exit map were
correct), A/B-benchmarked, and then REMOVED.** It LOSES wall-clock to plain
`pcg(A, fProxySSOR)` in every `@tol` cell measured, including the two cells where its iteration
count reached EXACT parity with plain SSOR-PCG — proof the loss is the per-iteration
arrangement itself (Algorithm 9.3's extra vector copies/adds and its verify-at-convergence
overhead), not an iteration-count artifact:

| section | dtype | plain SSOR-PCG (med ms / iters) | Eisenstat-PCG (med ms / iters) | iters equal? |
|---|---|---|---|---|
| b=1 stencil, N=10240 | float | 0.1700 / 2 | 0.2335 / 2 | yes — pure per-iter arrangement loss |
| b=1 stencil, N=10240 | double | 0.4211 / 5 | 0.4716 / 5 | yes — pure per-iter arrangement loss |
| BR=4, N=10240 | float | 2.1993 / 2 | 4.4696 / 6 | no (3x) |
| BR=4, N=10240 | double | 3.6458 / 3 | 10.3116 / 13 | no (4.3x) |

The BR=4 cells also exposed a real calibration bug en route (fixed before this verdict, kept
fixed below for the record): the internal transformed-space convergence gate was initially a
bare `tol²·‖r̂0‖²`, uncalibrated against the true `tol²·‖b‖²` criterion it stands in for; since
`r = (D̂−E)r̂` scales the two spaces by an uncontrolled operator-norm factor, this measured 2-4x
MORE iterations than plain SSOR-PCG needed on the BR=4/1.5%-fill instance. Rescaling the gate
by the r̂0/r0 ratio observed at entry (`thresholdHat = tol²‖b‖² · ‖r̂0‖²/‖r0‖²`, so the gate
targets the SAME relative reduction the true criterion targets) fully closed the gap on the b=1
stencil instance (exact iteration parity, both dtypes) but did NOT close it on the BR=4
instance (still 3-4.3x) — evidence the r̂/r scale ratio isn't constant across the whole solve
trajectory on that instance (a single entry-point calibration is a first-order fix, not an
exact one). Per-iteration cost WAS genuinely lower than plain SSOR-PCG's (float BR=4: ~0.74 vs
~1.10 ms/iter, ~32% cheaper, matching the "2 sweeps vs spMV+2 sweeps" theory) — but that saving
was smaller than the extra one-time setup/exit/verify overhead (visible in the stencil cells,
where iteration counts already matched) plus, on BR=4, the residual iteration-count penalty.

**Disposition**: removed in full — `Krylov.pcgEisenstat`, `fProxySSOR.ApplyEisenstat`, its
tests, and its `@tol` benchmark rows are all gone, "like it never shipped" (same treatment as
PDLP, `docs/pdlp-feature.md`). The derivation below is KEPT as a historical record — it is
correct and faithfully ported from Saad — should someone revisit this with a better-calibrated
or exact stopping criterion (e.g. Section "Avoiding the u-variable" below's per-iteration
proportionality assumption is the concrete thing to fix next, not the sweep math itself). Do
not revive without new evidence that the per-iteration/stopping-criterion overhead can be cut
below plain SSOR-PCG's.

**The round's other half — LOBPCG's SSOR preconditioner axis — is a KEEP**, unaffected by the
above (LOBPCG uses `fProxySSOR`'s plain `Apply`/TPre slot, not the removed split-preconditioned
Eisenstat path). Iterations AND wall-clock both improve sharply vs block-Jacobi, confirming the
round's hypothesis that LOBPCG's per-iteration cost is dominated by Rayleigh-Ritz work, not the
preconditioner apply — SSOR's larger apply cost (2-4x block-Jacobi, per R3) does not show up as
a wall-clock loss here the way it did in plain PCG:

| grid | dtype | none (ms/iters) | blockJac (ms/iters) | SSOR (ms/iters) | SSOR vs blockJac wall |
|---|---|---|---|---|---|
| 32x32 (1024) | float | 291.6/71 | 204.0/50 | 80.9/18 | 2.5x faster |
| 64x64 (4096) | float | 2541.5/126 | 1781.2/81 | 688.3/30 | 2.6x faster |
| 96x96 (9216) | float | 9048.9/178 | 6593.4/123 | 2591.0/38 | 2.5x faster |
| 32x32 (1024) | double | 580.6/170 | 410.2/115 | 148.7/38 | 2.8x faster |
| 64x64 (4096) | double | 5267.6/310 | 3678.3/220 | 1340.9/64 | 2.7x faster |
| 96x96 (9216) | double | 19777.2/464 | 14320.1/333 | 5374.3/96 | 2.7x faster |

SSOR consistently cuts LOBPCG iterations ~55-70% vs block-Jacobi (vs block-Jacobi's own ~30%
vs none) and wins wall-clock 2.5-2.8x vs block-Jacobi across every grid size and both dtypes —
the largest, cleanest win either preconditioner has produced in this spec's benchmarking. Kept
in `LargeSparseBenchmark`'s LOBPCG section (`none`/`blockJac`/`SSOR` x guard levers).

---

R3's own honest finding (`64b4431`): plain SSOR-PCG cuts iterations 33-67% vs block-Jacobi but
LOSES wall-clock in every `@tol` cell because `fProxySSOR.Apply` measures 2-4x block-Jacobi's
apply cost (two sequential triangular sweeps cannot pipeline across rows the way spMV can), not
the ~1 spMV-equivalent originally estimated. This section derives, from a fetched primary
source, the Eisenstat (1981) rearrangement that collapses SSOR-PCG's per-iteration cost to
**one forward sweep + one backward sweep total, with no separate `A·p` matvec** — the same
sweep budget plain SSOR-PCG already pays for `M.Apply` alone, but now it covers the matvec too.

**Source fetched**: Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (free PDF,
`www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf`), §9.2.2 "Efficient Implementations"
(Algorithm 9.3, the Eisenstat trick, pp. 280-281) for the core rearrangement, plus §10.2
"Jacobi, SOR, and SSOR Preconditioners" (p. 299, eq. 10.9/MSSOR) and ch. 4 eq. (4.27) for the
general-ω SSOR preconditioner form the trick is being fitted to. Cross-checked against the
existing `fProxySSOR.cs` doc comment's own independently-verified M formula (R3, `64b4431`).
Text extracted locally via `pdftotext -layout` (ω/η glyphs were dropped by the extractor in a
few spots — reconstructed from the surrounding equations and cross-checked against the
already-shipped, tested `fProxySSOR` formula, not reproduced blind).

### 9.2.2's setup (Saad's notation)

For symmetric A = D₀ − E − Eᵀ (−E the strict lower triangle, D₀ the diagonal) and a
preconditioner of the form M = (D−E)D⁻¹(D−Eᵀ) (eq. 9.6; D a diagonal, not necessarily D₀ —
SSOR with ω=1 is the D=D₀ special case), Eisenstat's implementation runs Algorithm 9.1
(standard PCG) on the transformed system

  Â u = (D−E)⁻¹b,  Â ≜ (D−E)⁻¹A(D−Eᵀ)⁻¹,  x = (D−Eᵀ)⁻¹u                     (9.7)-(9.8)

with the *extra* diagonal preconditioning M⁻¹ = D⁻¹ that Algorithm 9.1 must additionally apply
to reproduce the SAME iterates M=(D−E)D⁻¹(D−Eᵀ) would give directly. Expanding
Â = (D−E)⁻¹A(D−Eᵀ)⁻¹ using A = D₀−E−Eᵀ = (D₁) + (D−E) + (D−Eᵀ) with D₁ ≜ D₀−2D gives
Algorithm 9.3 for w = Âv:

  z := (D−Eᵀ)⁻¹v ;  w := (D−E)⁻¹(v + D₁z) ;  w := w + z

— one backward solve, one forward solve, no `A·v` matvec at all. Operation count (Nz = nonzero
count): Nop = 3n + 2Nz(A) for Eisenstat's scheme vs 4Nz(A) − n straightforward — "always more
economical when Nz is large enough" (Saad, Example 9.1: 5-point stencil, 23n vs 29n total
per-iteration ops including the rest of PCG).

### Fitting our ω-parameterized M to Saad's (D−E)D⁻¹(D−Eᵀ) form [this round's derivation]

`fProxySSOR`'s shipped M (R3, verified independently there) is
M = [ω/(2−ω)]·(D₀/ω+L)·D₀⁻¹·(D₀/ω+Lᵀ), L the strict-lower stored blocks (L = −E in Saad's
sign convention, Lᵀ = −Eᵀ). Substituting D̂ ≜ D₀/ω (so D₀ = ωD̂):

  M = [ω/(2−ω)]·(D̂−E)·(ωD̂)⁻¹·(D̂−Eᵀ) = [1/(2−ω)]·(D̂−E)·D̂⁻¹·(D̂−Eᵀ)

which is exactly Saad's (D−E)D⁻¹(D−Eᵀ) form with D := D̂ = D₀/ω, up to the scalar 1/(2−ω) —
and scalar prefactors on M are provably irrelevant to PCG (next paragraph), so they can be
dropped. This means **every `BSR.sweepLower`/`sweepUpper` call already used by `fProxySSOR`
with `diagScale=ω` is exactly Saad's (D̂−E)⁻¹ / (D̂−Eᵀ)⁻¹ solve** — no new sweep kernel needed —
and D₁ = D₀ − 2D̂ = D₀(1 − 2/ω) = −[(2−ω)/ω]D₀ = **−ScaledD**, the diagonal `fProxySSOR`
already precomputes at construction for the plain `Apply` path. So "v + D₁z" = v − ScaledD·z,
computable with the existing private `ApplyScaledDiag` helper — Algorithm 9.3 falls out of
pieces `fProxySSOR` already owns.

**Lemma (PCG is invariant to a positive scalar on M — used twice above)**: replacing M by cM
(c>0) in Algorithm 9.1 leaves the x_j and r_j sequences bit-for-bit unchanged; only z_j,p_j
scale by 1/c (induction on the standard PCG recurrence: αⱼ scales by c exactly canceling
z_j'/p_j' = z_j/c, p_j/c in the x/r updates; βⱼ = ⟨r,z'⟩/⟨r,z'⟩ is scale-free). This is the
formal version of Saad's own remark (ch. 4, on M_SOR/M_SSOR's leading constants) that "these
constant coefficients... are unimportant in the preconditioning context." It licenses BOTH
dropping the 1/(2−ω) prefactor above AND — since M⁻¹=D̂⁻¹=ωD₀⁻¹ is *also* just a positive
scalar (ω) times D₀⁻¹ — replacing the "extra diagonal preconditioning" Saad calls for with
**`fProxyBlockJacobi.Apply` unchanged** (D₀⁻¹, not ωD₀⁻¹): same x_j/r_j sequence, zero new
kernel, direct reuse of `fProxySSOR.Jacobi` (already "block-Jacobi's own setup" per the R3 doc
comment).

### Avoiding the u-variable, and the convergence-test problem it creates

Per Saad's own earlier remark in §9.1 (deriving Algorithm 9.2 from 9.1): "It is common when
implementing algorithms which involve a right preconditioner to avoid the use of the u
variable, since the iteration can be written with the original x variable." Applied here: track
ũⱼ ≜ uⱼ−u₀ (so ũ₀=0, no need to ever materialize u₀ itself — r̂₀ = (D̂−E)⁻¹r₀ already encodes
it, since r̂₀ = ĉ−Âu₀ = (D̂−E)⁻¹(b−Ax₀) by (9.7)-(9.8), independent of how u₀ itself is chosen).
ũ accumulates exactly like Algorithm 9.1's own u-update (ũⱼ₊₁ = ũⱼ+αⱼp̂ⱼ); the map back is
**one extra sweep AT EXIT**: x = x₀ + (D̂−Eᵀ)⁻¹ũ_final (`BSR.sweepUpper`), not per iteration —
matching the task brief's framing exactly.

This has one consequence Saad's chapter doesn't dwell on: the loop's own residual r̂ⱼ lives in
*transformed* space (r̂ⱼ = (D̂−E)⁻¹rⱼ, an exact invariant of the recurrence, provable by the same
induction as the lemma above), not the TRUE rⱼ = b−Axⱼ this library's other solvers test
against (`docs/draft-spec-krylov-optimization.md` §0/pcg's own contract: "the TRUE
(unpreconditioned) residual"). Recovering true rⱼ every iteration would need a triangular
MATVEC (not solve) by (D̂−E) — a third sweep-equivalent op per iteration, which would erase
Eisenstat's entire saving. Two facts make a cheap resolution possible:

1. In exact arithmetic the xⱼ sequence produced here is IDENTICAL, iteration for iteration, to
   plain `pcg(A, fProxySSOR, ...)`'s xⱼ (both run Algorithm 9.1 on the same M, just in
   different coordinates — this is the same equivalence Saad proves between Algorithms 9.1 and
   9.2). So any correct stopping rule that fires near where the true criterion would is enough;
   it does not need to BE the true criterion every step.
2. R3's `MakeSolveInfo` doc comment already carries an approved escape hatch for exactly this
   class of problem: "Convergence-verification matvec: APPROVED. +1 Apply at claimed
   convergence is an accepted amendment to the never-a-fresh-A·x diagnostics contract" (R6a,
   RESOLVED QUESTIONS §1).

**Design**: the loop's cheap internal gate is the natural Algorithm-9.1-on-(Â,ĉ) criterion,
‖r̂ⱼ‖² ≤ thresholdHat (mirrors `cg<TOp>`'s own "test against a fixed initial scale" shape, just
computed on the transformed right-hand side instead of `b` — no new concept, reuses
`Blas.updateXR` for the fused ũ/r̂ update+norm exactly as plain `cg`/`pcg` already do for x/r).
thresholdHat is NOT bare `tol²·‖r̂₀‖²` — r̂ and r live in different scales (r = (D̂−E)r̂, an
uncontrolled operator-norm factor), and an unscaled transformed threshold measured 2-4x MORE
iterations than plain SSOR-PCG needs on some benchmark instances (a real efficiency bug caught
by A/B benchmarking, not just a "few extra iterations" — see the R3b benchmark report).
thresholdHat is instead calibrated by the r̂₀/r₀ ratio actually observed at entry (both already
computed there): `thresholdHat = threshold · (‖r̂₀‖²/‖r₀‖²)`, so the gate targets the SAME
relative reduction ‖r̂ⱼ‖²/‖r̂₀‖² ≈ ‖rⱼ‖²/‖r₀‖² the true criterion targets, rather than an
uncalibrated absolute scale. Re-calibrated the same way after every restart (below), using the
just-verified true residual as the new reference point. When the gate fires: do the exit sweep
(map ũ→x), then (R6a) one real `A.Apply` + dot to get
the TRUE ‖b−Ax‖² and check it against the library-standard tol²·‖b‖². If it passes, return
Converged with the FRESH true rnorm. If it fails (expected to be rare, given point 1 above):
treat the just-computed true residual as a fresh restart point — reset ũ:=0, rebuild
r̂/ẑ/p̂/the gate threshold from it via the same setup the entry path uses, and continue the outer
loop. This is a bounded, self-correcting design: the OUTPUT contract (`Converged` iff the true
residual test passes) is identical to plain `pcg`'s, regardless of how well-calibrated the
internal gate turns out to be; a miscalibrated gate only costs a few wasted real `Apply`s, never
an incorrect answer. Breakdown/MaxIterations exits still perform the mandatory exit sweep (x
must never be returned unmapped) but skip the extra real `Apply` (matches the "no fresh matvec
outside claimed convergence" contract everywhere else — they report the last-verified true
rnorm, a value the solver already holds, same as `cg`/`pcg`'s own breakdown/max-iterations
`rnorm` fields).

### Net per-iteration cost

One `fProxySSOR.ApplyEisenstat` call (`sweepUpper` + diagonal scale + `sweepLower` + vector
adds — exactly Algorithm 9.3, using `Scratch1`/`Scratch2` already on the struct), one
`Blas.dot` (αⱼ), one `Blas.updateXR` (ũ/r̂ update + ‖r̂‖² fused), one `fProxyBlockJacobi.Apply`
(ẑ, diagonal-only — O(nnzb_diag), not a full sweep), one `Blas.dot` (βⱼ). Two sweeps total, zero
separate `A·p`, matching the task brief's target exactly — the same sweep budget plain SSOR-PCG
already pays for `M.Apply` alone now also covers what used to be a separate spMV.

---

## 4. References

- Eisenstat, "Efficient implementation of a class of preconditioned conjugate gradient
  methods", SIAM J. Sci. Stat. Comput. 2(1), 1981 — the SSOR trick (R3 follow-up: makes
  SSOR-PCG's per-iteration cost ≈ unpreconditioned CG's; adopt only if R3's plain form
  measures well). [verified via epubs.siam.org/doi/10.1137/0902001]
- **R3b (2026-07-10) [verified, fetched and read directly, see §3b above for the full
  derivation]**: Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., free PDF at
  `www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf` — §9.2.2 "Efficient Implementations"
  (Algorithm 9.3, "Eisenstat's implementation"/"Eisenstat's trick", pp. 280-281, Example 9.1)
  for the core rearrangement; §9.1 (deriving Algorithm 9.2 from 9.1, the "avoid the u
  variable" remark used for the exit-map design) pp. 277-279; §10.2 "Jacobi, SOR, and SSOR
  Preconditioners" p. 299 (eq. 10.9, MSGS=MSSOR(ω=1)) and ch. 4 eq. (4.27) (general-ω MSSOR)
  for the preconditioner form Eisenstat's trick is fitted to here. Extracted locally via
  `pdftotext -layout` (poppler-utils) since the WebFetch tool cannot parse this PDF's compressed
  stream directly.
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
