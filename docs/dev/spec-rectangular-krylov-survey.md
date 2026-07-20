# Survey: rectangular / least-squares / least-norm Krylov landscape

Status: RESEARCH / SURVEY (no solver code here). Question: what does LinearAlgebraBursted already
ship for non-square Krylov solves, what is the full landscape of methods for that problem family,
and what should be built next, in what order. This document is a map for future single-session
specs, not an implementation spec itself -- signatures below are scoping sketches, not final APIs.

## 1. Inventory: what we already ship

### 1.1 The lsqr / lsmr overload ladder

Both `lsqr<TOp>` and `lsmr<TOp>` converge on ONE damped generic core each; every other overload is a
thin forwarding wrapper. All in `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.fProxy.cs`:

| Core / wrapper | lsqr line | lsmr line |
|---|---|---|
| damped generic core `<TOp>` | 1103 | 1419 |
| undamped forward `<TOp>` | 1265 | 1599 |
| dense zero-alloc (`fProxyMxN`, caller scratch) | 1276 | 1610 |
| dense allocating, undamped | 1285 | 1619 |
| dense allocating, damped | 1299 | 1634 |
| dense, default maxIter/tol | 1310 | 1646 |
| BSR zero-alloc (builds `AT` on the fly per ApplyT) | 1322 | 1657 |
| BSR zero-alloc, caller-supplied `AT` | 1341 | 1673 |
| BSR allocating undamped (auto-builds `AT` once) | 1358 | 1688 |
| BSR allocating damped | 1374 | 1705 |
| BSR, default maxIter/tol | 1386 | 1718 |

Plus `lstsqResidual` (line 1054, a certified `LstsqInfo` auditor: one fresh `Apply`+`ApplyT`) and the
Jacobi convenience layer (`lsqrJacobi`/`lsmrJacobi`, dense+BSR+default, lines 1751-1841). That is ~31
public entry points funneling into 2 numerical cores -- a healthy "many doors, one room" ladder, no
duplicated recurrence logic to consolidate.

### 1.2 The transpose contract

`IfProxyLinearOperator` (`Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/LinearOperator.fProxy.cs:15-45`)
requires both `Apply` (y=Ax) and `ApplyT` (y=A^T x) -- there is no transpose-free rectangular solver
path (contrast `biCGStab`, which the interface doc explicitly calls out as never touching `ApplyT`,
same file line 24). `fProxyDenseOperator.ApplyT` (line 91) reuses the existing vector-matrix dot
kernel, no separate transpose materialization. `fProxyBSROperator`
(`Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxyBSROperator.cs`) has two modes: a one-arg
ctor where `ApplyT` runs an on-the-fly scatter-traversal `spMVT` every call (cheap to build, expensive
and cache-unfriendly per call -- `docs/features/sparse-bsr.md:53` attributes the ~7-8x vs. an ideal
~14x sparse speedup gap directly to this), and a two-arg ctor taking a precomputed `AT` where `ApplyT`
forwards to a cache-friendly forward `spMV(AT, x)`. Every BSR "allocating" overload without an explicit
`AT` (e.g. `lsqr(in fProxyBSR, in fProxyN, ref fProxyN, int, fProxy)` at line 1358) calls
`b.fProxyTempVec`/`b.fProxyBSRTranspose(in A)` once per solve -- so it is zero-alloc *per iteration*
but not zero-alloc *per call*; the true zero-alloc primitive is the caller-supplied-`AT` overload
(line 1341/1673).

### 1.3 How `damp` threads through

Both solvers fold a scalar Tikhonov `damp` into their Givens-rotation state per iteration (lsqr:
`rhobar1` at lines 1198-1208; lsmr: `alphahat`/`(chat,shat)` at lines 1526-1533) rather than running a
separate damped code path -- `damp == 0` is documented bit-identical to the undamped solve. Both carry
an explicit **warm-start + damping gotcha** in their doc comments (lsqr lines 1095-1097, lsmr lines
1410-1413): because Golub-Kahan bidiagonalizes the residual `b - A.x0`, a nonzero `x0` makes damping
regularize the *correction* `||x-x0||`, not `||x||` -- a property intrinsic to bidiagonalization-based
damping, not a bug, and one that CRAIG's damped extension (SS4a) inherits identically.

### 1.4 The existing "LS Jacobi" IS a right-preconditioner

`lsqrJacobi`/`lsmrJacobi` are not a bolt-on `M^-1`-style wrapper; they already do literal RIGHT
preconditioning: build `D = diag(1/||A_:,j||)` (`Blas.columnNormsSquared` + `Blas.buildJacobiScale`),
wrap `A` as `fProxyColScaledOperator<TInner>` presenting `A.D`
(`Interfaces/LinearOperator.fProxy.cs:167-231`), solve `(A.D) y = b` for `y` with the plain generic
`lsqr<TOp>`/`lsmr<TOp>`, then unscale `x = D.y`. This is exactly the "solve for `A.N^-1`, recover `x`"
shape described in SS3 -- the mechanism already exists, it is just hard-wired to the Jacobi-derived
diagonal `D` and cold-start only (no `damp`/warm-start parameter: line ~1728's comment notes column
scaling is a change of variable, so a warm `x0` would need a `y0 = D^-1 x0` pre-map that is not wired
up). Diagnostics are recomputed in *original* coordinates post-solve via `lstsqResidual`
(`JacobiFinish`, line 1751) because the tracked `Arnorm` during the scaled solve is `||D.A^T r||`, not
the true `||A^T r||`.

### 1.5 Public docs vs. code: a mismatch to flag

`docs/features/least-squares.md` (lines 22-23, 41, 45-46) and `README.md:43` currently document
`cgls`/`cglsJacobi`/`cgne` (Craig's method) as shipped APIs. Per
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/DEVLOG.md` ("Krylov.cgne removed", "Krylov.cgls /
cglsJacobi removed"), **both were deliberately deleted** -- `cgne` on 2026-07-18, `cgls`/`cglsJacobi`
on 2026-07-19 -- because both are normal-equations methods (kappa-squared conditioning) dominated by
`lsqr`/`lsmr`, which solve the identical problems (including the Tikhonov `damp` and Jacobi wrapper) at
strictly better conditioning. The public docs were not updated in the same pass and now describe a
**non-existent API**. This is out of scope to fix inside this survey (no solver-code or README edits
here per the task brief), but it should be corrected before any of Section 4's new methods ship, so the
next doc pass does not have to disentangle stale-vs-new claims about the same file.

## 2. The full rectangular/least-norm Krylov landscape

| Method | Problem | Recurrence family | Needs A^T | Monotonicity | Status | Verdict |
|---|---|---|---|---|---|---|
| LSQR | overdetermined `min||Ax-b||` (any shape) | Golub-Kahan bidiag + incremental Givens QR on the bidiagonal | yes | `||r||` non-increasing; `||A^T r||` not guaranteed monotone | **HAVE** | **KEEP** (Section 7) |
| LSMR | same problem | same bidiag, MINRES-folded onto the normal equations | yes | `||A^T r||` monotone (its whole selling point) | **HAVE** | **KEEP** -- preferred default (Section 7) |
| CGLS/CGNR | overdetermined LS | CG directly on `A^T A x=A^T b`, `A^T r` recomputed fresh each iter (avoids CGNR residual drift) | yes | monotone in the `A^T A`-norm | **REMOVED** (2026-07-19, deliberate -- kappa^2, dominated) | REPLACED (already executed, by lsqr/lsmr) |
| CGNE | underdetermined least-norm, CG on `A A^T` | normal equations on `A A^T`, no bidiagonalization | yes | monotone in `A A^T`-norm | **REMOVED** (2026-07-18, deliberate -- kappa^2, redundant with lsqr's min-norm property) | REPLACED (already executed, by lsqr) |
| **CRAIG** | underdetermined least-norm `min||x|| s.t. Ax=b` (consistent) | **same** Golub-Kahan bidiag as lsqr/lsmr, but forward-solves the lower-bidiagonal `Lk yk=beta1 e1` directly -- no Givens-rotated LS subproblem needed when undamped | yes | monotone `||xk||` growth; SYMMLQ-like properties | **GAP -- primary** | n/a (not yet built) |
| CRAIGMR | damped/inconsistent least-norm (CRAIG + MINRES-style residual tracking, mirrors how LSMR relates to LSQR) | same bidiag | yes | monotone dual-residual quantity | **GAP** | n/a (not yet built) |
| LSLQ | overdetermined LS, SYMMLQ-analogue of LSQR | same bidiag, SYMMLQ-folded | yes | error-norm bound, not residual | GAP, low priority | n/a -- reconsidered in Section 7.2, does not change the LSQR/LSMR verdict |
| LNLQ | underdetermined least-norm, SYMMLQ-analogue of CRAIG | same bidiag, SYMMLQ-folded on the dual | yes | error-norm bound | GAP, low priority | n/a (not yet built) |
| Generalized Tikhonov `[A;lambda L]` (`L != I`) | `min||Ax-b||^2+lambda^2||Lx||^2`, seminorm regularization | same bidiag over a **stacked** operator (`Rows=m+p`) | yes, `A^T_stacked=[A^T | lambda L^T]` | inherits base solver's monotonicity | GAP (only scalar `damp` == `L=I` shipped) | n/a (not yet built) |
| Right/split preconditioning (general `N`, not just diagonal Jacobi) | conditioning fix for either family | `A.N^-1` composition, same bidiag | yes | unchanged | **PARTIAL** -- mechanism (`fProxyColScaledOperator`) exists, not generalized past diagonal-Jacobi | n/a (not yet built) |
| Block-LSQR/LSMR | multi-RHS overdetermined LS, `s` RHS sharing one Krylov subspace | block Golub-Kahan (`U_k`,`V_k` become `s`-wide) | yes, batched | inherits per-column | GAP | n/a (not yet built) |

**Code-reuse callout:** Golub-Kahan bidiagonalization is the *single shared engine* behind
LSQR/LSMR/LSLQ/CRAIG/CRAIGMR/LNLQ. Every currently-missing method in this table (except block-LSQR,
which needs a block-*shaped* bidiagonalization, not a different one) can reuse the exact `u`/`v`
generation loop already proven correct and Burst-fast at lines 1187-1196 (lsqr) / 1512-1523 (lsmr) --
the missing pieces are alternative ways to *fold* that same `(alpha, beta)` sequence into a solution,
not a new matrix-free kernel.

## 3. Why square preconditioning does not map onto rectangular LS/LN Krylov

For a square SPD system, `M^-1` on the LEFT preserves the solve's structure: PCG implicitly solves
`M^-1 A x=M^-1 b` in the `M`-inner-product and stays symmetric provided `M` is SPD. For
LSQR/LSMR/CRAIG, `A` is `m x n` and the quantity being minimized is `||Ax-b||` itself (or `||x||` s.t.
`Ax=b`) -- there is no `M^-1 A` available on the left that keeps the *same* objective, because `M`
would have to be `m x m` (matching `Rows`), and `min||M^-1(Ax-b)||` is a **different,
residual-reweighted** least-squares problem, not a preconditioned iteration converging to the same
answer. Left "preconditioning" for rectangular LS is really *row weighting* -- a legitimate but
distinct tool (e.g. downweighting noisy observations), not the conditioning fix for slow
bidiagonalization convergence, which is governed by `A`'s own singular value spread.

The correct lever is a **right** preconditioner `N` (`n x n`, matching `A`'s column space): solve
`(A.N^-1) y = b` for `y`, recover `x = N^-1 y`. This reshapes `A`'s singular values without changing
the objective at all -- `||(A.N^-1)y-b|| = ||Ax-b||` exactly under `x=N^-1 y`, unlike left-weighting.
This is *precisely* what `fProxyColScaledOperator` already implements for the diagonal case `N^-1=D`
(Section 1.4). A **split/two-sided** scheme (row AND column scaling, i.e. Ruiz equilibration -- already
used for the LP IPM path, see `research-lp-preconditioners.md`) is the natural generalization for the
harder ill-scaled BSR cases, and is worth reusing rather than reinventing.

**Regularization is a related but different lever.** `damp` changes the *effective* conditioning the
bidiagonalization has to fold through, but it also changes the answer (the Tikhonov-regularized
solution != the original LS/LN solution) -- so it is not "preconditioning without changing the answer"
the way right-preconditioning is. Generalized Tikhonov `[A;lambda L]` is a way to make that deliberate
answer-change *smarter* (penalize a seminorm `||Lx||` instead of `||x||`), not a way to avoid it.

**Interface recommendation:** `IfProxyPreconditioner` (the `M^-1 r`-shaped interface used by
`pcg`/`pminres`/`pbiCGStab`) does **not** fit rectangular right-preconditioning -- `N` here is `n x n`
and gets *composed with `A`*, not applied independently to a residual vector. No new interface is
actually needed: the codebase's existing convention (`fProxyColScaledOperator<TInner>`,
`fProxyNormalOperator<TInner>` in `Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/fProxySparseLP.fProxy.cs:39-72`)
is "wrap a `TInner : struct, IfProxyLinearOperator` in a new struct implementing the same interface,
reuse the generic solver unchanged." Concretely, task #21 ("generalize LS Jacobi into a right
preconditioner") most likely resolves as: **delete the bespoke duplication**, not add an interface --
`lsqrJacobi`/`lsmrJacobi` become thin callers of the *already-generic* `lsqr<TOp>`/`lsmr<TOp>` with a
`fProxyColScaledOperator`-shaped `TOp`, and `JacobiFinish`'s "unscale + re-diagnose in original
coordinates" logic generalizes from "Jacobi-derived `d`" to "any `fProxyColScaledOperator`-shaped
wrapper" (it already only needs the wrapper's `D` field and the unscaled inner operator -- nothing
Jacobi-specific). This lets a caller supply a *custom* `d` (not just column-norm-derived) through the
same path for free, and is the natural place to also land task #20 (Section 4b).

## 4. Gap analysis + prioritized roadmap

### (a) CRAIG / CRAIGMR -- value HIGH, effort LOW-MEDIUM -- recommended first target
Fills the *only* completely unaddressed problem class (least-norm), reusing the exact
Golub-Kahan loop already shipped and proven. Per Paige & Saunders' own framing (BIT 1995, abstract):
undamped CRAIG is "slightly simpler and more efficient" than LSQR because it forward-solves a
lower-bidiagonal system instead of running a Givens-rotated least-squares subproblem -- so this is not
just "a new solver," it is arguably a *cheaper* recurrence than what is already shipped. Spec sketch
for the follow-up implementation session: new file `OP/Krylov.LeastNorm.fProxy.cs`; a
`craig<TOp>(in A, in b, ref x, ref u, ref v, ref <TBD w/accumulator>, int maxIter, fProxy tol)` generic
core sharing lsqr's `u`/`v` bidiagonalization step exactly, but with its OWN solution-accumulation
recurrence (do not guess it from this survey -- derive/verify against `craigSOL.m`, since the
lower-bidiagonal forward-solve update is structurally different from lsqr's Givens-`w` update); a
right-sized overload ladder (generic + dense zero-alloc + dense allocating + BSR zero-alloc-with-`AT` +
BSR allocating -- probably NOT all 11 lsqr-style rungs for a v1). CRAIGMR (the damped/inconsistent
extension, Section 2's BIT-1995 SS4.4 "Extended CRAIG") is a natural phase 2 once the phase-1
recurrence is proven, since it reuses the same extended-bidiagonalization idea the paper already works
out.

### (b) Generalize right-preconditioner (#21) + `[A;lambda L]` stacked operator (#20) -- value MEDIUM-HIGH, effort LOW
Pure composition, no new bidiagonalization risk -- both are "wrap `A` as a new `IfProxyLinearOperator`,
reuse the existing generic solver." Natural to do together: `[A;lambda L]` is itself expressible as
"wrap `A` as a stacked operator" the same way Jacobi wraps `A` as a column-scaled operator. Spec
sketch: `fProxyStackedOperator<TA,TL>` (in `Interfaces/LinearOperator.fProxy.cs`, alongside
`fProxyColScaledOperator`): `Rows = TA.Rows + TL.Rows`, `Cols = TA.Cols`; `Apply` writes into two
separate caller-provided output vectors (`yTop` length `TA.Rows`, `yBot` length `TL.Rows`) rather than
one concatenated buffer -- check whether `fProxyN` has a zero-copy sub-range view before assuming a
single-buffer split is free; per the backlog, View/Slice work is deliberately last, so **do not**
block this on that item, just use two full vectors composed by the caller if no cheap slice exists.
`ApplyT` sums `A^T(yTop) + lambda.L^T(yBot)` into one `n`-vector (`lambda` as a scalar field on the
struct, mirroring `fProxyNormalOperator`'s `Reg` field). The right-preconditioner generalization is
described in Section 3 above -- expect it to land as reduced/deleted code in
`lsqrJacobi`/`lsmrJacobi`, not new surface area.

### (c) Block-LSQR / block-LSMR -- value MEDIUM, effort MEDIUM-HIGH
Needs a genuine *block* Golub-Kahan bidiagonalization (`U_k`/`V_k` become `s`-wide blocks,
orthogonalized via the block Gram + Cholesky idiom `bcg`/`bcgrq` already use in
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.fProxy.cs`, e.g. `BlockGram`/
`BlockSolveSPD` at lines 21-79). The one real interface obstacle: `IfProxyLinearOperator.ApplyBlock` is
explicitly documented symmetric-operator-only today (`Interfaces/LinearOperator.fProxy.cs:36-44`,
"the only caller ... always passes symmetric A") -- a rectangular `A`'s block-apply needs an
input-block-width-`Cols`/output-block-width-`Rows` shape that the current single-shape `ApplyBlock`
does not support, so this touches the *operator interface*, not just composition, and should be scoped
as its own sub-spec once (a)/(b) land. `OP/DEVLOG.md` (~line 129) already records a "for the future
BLOCK-Krylov rewrite" plan for 7 unified block algorithms -- folding block-LSQR/LSMR into that plan
(rather than bolting it on separately) avoids a second, incompatible block-vector convention.

### (d) LSLQ / LNLQ -- value LOW, defer indefinitely
Their selling point is a provable *error-norm* bound / smoother a-priori stopping criterion, valuable
for genuinely ill-posed inverse problems with a target error tolerance. This library's actual workload
(per `docs/features/sparse-bsr.md`'s benchmark numbers) is well-posed-ish sparse LS/LN over BSR
Jacobian-like operators, where LSQR/LSMR's residual-based stopping already suffices. Noted here only so
a future pass does not have to rediscover they exist.

## 5. Reference implementations + licensing

Ports must come from **permissive** sources -- the library is heading to public v1.0 with an existing
GPL-taint cleanup item, and algorithms themselves are not copyrightable, so the rule is: permissive
sources may be read closely and mirrored structurally; copyleft sources are algorithm-description-only,
reimplemented from the math, never read side-by-side while writing the port.

| Method | Primary reference | License | Use |
|---|---|---|---|
| LSQR | Paige & Saunders, TOMS 1982 (`reference/rectangular/LSQR-Paige-Saunders-TOMS1982.txt`) | open access | already shipped -- no action |
| LSMR | Fong & Saunders 2010/2011 (`reference/rectangular/LSMR-Fong-Saunders-2010.txt`) | open access | already shipped -- no action |
| CRAIG / CRAIGMR | `craigSOL.m` (Stanford SOL software page -- fetch current URL at implementation time; historically `web.stanford.edu/group/SOL/software/craig/`) | permissive research software (SOL "free software") | **primary port target** |
| CRAIG / CRAIGMR (derivation) | Paige & Saunders, BIT 1995 (`reference/rectangular/CRAIG-Paige-Saunders-BIT1995.txt`, Section 4.4 "Extended CRAIG Algorithm") | open access | algorithm derivation/proof |
| CRAIG / CRAIGMR (iteration-loop idiom) | Belos `LSQRIter` (`reference/rectangular/BelosLSQRIter.hpp`, Trilinos) | **BSD-3-Clause** | check the same BSD-3 Trilinos tree for a `Belos::*Craig*` iteration class before assuming none exists; otherwise use only for the iterate/status-test loop shape, not CRAIG-specific math |
| CRAIG / CRAIGMR (cross-check only) | Krylov.jl `craig.jl`/`craigmr.jl` (JuliaSmoothOptimizers) | **MPL-2.0** (file-level weak copyleft) | algorithm cross-check only -- never copy source |
| `[A;lambda L]` generalized Tikhonov | direct consequence of shipped `damp` + bidiagonalization; Bjorck, *Numerical Methods for Least Squares Problems* (textbook) for the standard-form reduction | n/a (no file fetch needed) | conceptual reference only |
| Block-LSQR/LSMR | Karimi & Toutounian 2008, "The block least squares method for solving nonsymmetric linear systems with multiple right-hand sides" (journal, algorithm-only) + in-repo `bcg`/`bcgrq` block-Gram/block-solve idiom | n/a | no permissive code found to port; derive from the paper's equations + our own block idiom |
| LSLQ / LNLQ (if ever revisited) | Krylov.jl `lslq.jl`/`lnlq.jl` | **MPL-2.0** | algorithm-only; Fong-Saunders LSMR paper's SYMMLQ discussion (already fetched) explains the LSQR:LSLQ::LSMR:MINRES duality well enough to avoid a dedicated new paper fetch |

Do not introduce any new file whose structure was derived by close reading of an MPL/GPL source --
Krylov.jl is reference-only for verifying a recurrence is *correct*, never for phrasing or structure.

## 6. Project constraints (recap for the implementing session)

- Templates under `Assets/LinearAlgebra/CodeGen/TemplateSource*` are the source of truth; never edit
  `Assets/LinearAlgebra/Source/Generated` directly.
- `fProxy` is the codegen dtype token (float/double/int per generation); row-major `fProxyMxN`; any
  block convention (Section 4c) is `s` rows x `n` cols, matching `bcg`/`bcgrq`'s existing convention
  (`Krylov.Block.fProxy.cs:12-14`).
- Comments state contracts only (what a member requires/destroys/returns); algorithm rationale,
  derivation notes, and rejected alternatives go in the folder's `DEVLOG.md`, never in code comments.
- Short parameter names throughout: `maxIter`, `tol`, `damp` (library-wide ruling -- never expand to
  `maxIterations`/`tolerance`/`damping`).
- Zero-alloc discipline matches lsqr/lsmr: a generic `<TOp>` core takes caller-provided scratch vectors
  and never allocates; convenience overloads allocate once from the Arena via `b.fProxyTempVec`/
  `b.fProxyBSRTranspose`.
- Do not touch `README.md` (per this survey's brief). The stale `cgls`/`cgne` references in
  `docs/features/least-squares.md` (Section 1.5) are a separate, already-flagged fix, not part of this
  deliverable.
- This document is a survey: no solver code, no template edits, deliverable is this file.

## 7. State-of-the-art replacement audit: LSQR and LSMR

Precedent for this lens: the block-Krylov work found that ad-hoc ridge-regularized block-CG is best
replaced by BFBCG (Ji & Li 2017), a properly-named, breakdown-free method with no ridge and no
normal-equations kappa^2 (`reference/papers/BFBCG-algorithm-extract.md`). That is a genuine REPLACE
case: BFBCG dominates the ad-hoc approach on every axis (stability, cost, provenance) with nothing
given up. Applying the same lens here: **neither LSQR nor LSMR is an ad-hoc method** -- both are
themselves the named, peer-reviewed, still-actively-shipped-by-every-major-library (SciPy, PETSc,
Trilinos/Belos) state of the art for their problem. The question is narrower than the block-CG case:
not "is there a better-established method to replace an ad-hoc one," but "given we ship both, which
is the right default, and does anything post-2010 dominate either." Verdict up front: **KEEP both,
no retirement** -- unlike CGLS/CGNE (Section 2), LSQR and LSMR are not dominated by each other in the
CGLS/CGNE sense (kappa^2-worse, redundant); they are two different foldings of the *same*
bidiagonalization, each with a genuine, literature-documented use case. The actionable finding is
about our **stopping-test wiring**, not about deleting a solver.

### 7.1 Is LSMR strictly dominant over LSQR? -- KEEP both, but fix an asymmetry in how we use them

Fong & Saunders' own paper (`reference/rectangular/LSMR-Fong-Saunders-2010.txt`) is precise about
this, and it is worth quoting because it changes the recommendation from "roughly similar" to
"concretely asymmetric":

- Line 763: `rLSQR is monotonic by design. rLSMR seems to be monotonic (no counter-...)` -- i.e. LSQR's
  own **residual** `||r||` (`phibar` in our implementation) is guaranteed monotonically decreasing by
  design, exactly as advertised. LSMR's `||r||` is *empirically* monotonic (not proven) but, per lines
  9-11, "never very far behind the corresponding value for LSQR."
- Lines 9-11: `...ATrk are monotonically decreasing... compared to LSQR (for which only rk is
  monotonic) it is safer to terminate LSMR early.`

So the literature's own framing is: **LSQR's strength is a provably monotonic `||r||`; LSMR's strength
is a provably monotonic `||A^T r||`.** Neither dominates the other in the abstract -- they guarantee
monotonicity of *different* quantities.

**The concrete finding for this codebase:** our `lsqr<TOp>` does NOT use its own strength as the
stopping test. Both `lsqr` and `lsmr` (`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.fProxy.cs`,
lines 1231 and 1587) stop on the identical criterion, `arnorm*arnorm <= threshold` where
`arnorm = ||A^T r||`. For `lsmr` this is exactly right -- `zetabar` (its `arnorm`) is the solver's own
proven-monotonic quantity. For `lsqr`, `arnorm` (`phibar*alpha*|c|`, line 1229) is the **non-monotonic**
quantity per the paper above; `lsqr` also tracks `phibar` (`||r||`, or the damping-augmented residual,
line 1223), which per the same paper IS monotonic by design -- and that quantity is *not* used in the
stopping test at all, only reported afterward via `LstsqInfoTracked`. In other words: our `lsqr` is
implemented with LSMR's stopping-test philosophy grafted onto LSQR's recurrence, discarding the one
guarantee LSQR actually offers. This does not make `lsqr` incorrect (a value below the arnorm threshold
at iteration k is still a real, checkable fact about x_k), but it means our `lsqr` gets none of LSQR's
textbook safety advantage over `lsmr` while (per 7's Fong-Saunders quote) `lsmr`'s `||r||` is "never very
far behind" LSQR's anyway -- so as currently wired, `lsqr` has no remaining edge over `lsmr` for this
library's purposes.

**Recommendation:**
- **LSMR: KEEP, unchanged, and make it the recommended default** for new call sites / any future
  facade work (the `regression-fitting-api-organization` backlog item, when it lands, should default
  to `lsmr`, not `lsqr`). Its stopping test already uses its own proven-monotonic quantity; nothing to
  fix.
- **LSQR: KEEP** (do not retire -- it is CRAIG's/CRAIGMR's natural sibling: Section 4's recommended
  first target, CRAIG, is structurally closer to LSQR's forward-substitution style than to LSMR's
  MINRES-folded style, so deleting LSQR would not even save implementation work for CRAIG; LSQR also
  remains the more commonly cited baseline in the wider literature for reproducing published results).
  **AUGMENT candidate, not required for this survey to resolve:** either (a) switch `lsqr`'s stopping
  test to check its own monotonic `phibar`-based residual (with a corresponding `tol` reinterpreted as
  a relative-residual tolerance, `phibar <= tol*||b||`, rather than the current `A^T r`-based one --
  a **behavior change**, needs its own spec deciding whether to keep both criteria as an "either"
  check or fully switch), or (b) leave `lsqr` as-is and simply document in `docs/features/least-squares.md`
  that `lsmr` is the safer default and `lsqr`'s early-termination guarantee is on `||r||`, not `||A^T
  r||`, despite the code checking the latter. Either is a small, separately-specced follow-up; flagging
  it here so it is not lost, not resolving it in this survey.

### 7.2 Post-2010 developments: hybrid/flexible LSQR, LSLQ/LNLQ reconsidered

**Hybrid projection methods** (Chung, Nagy, O'Leary and successors; comprehensively surveyed in Chung &
Gazzola, "Computational Methods for Large-Scale Inverse Problems: A Survey on Hybrid Projection
Methods," SIAM Review, ~2024) project the problem onto the *same* growing Golub-Kahan subspace LSQR/LSMR
already build, but solve a small regularized subproblem at each iteration and pick the Tikhonov
parameter *adaptively* (via GCV, the L-curve, the discrepancy principle, or an unbiased predictive risk
estimator) instead of the caller fixing a scalar `damp` up front, as our `damp` parameter requires today.
"Flexible" variants (FLSQR) additionally vary the *preconditioner* every iteration to support
edge-preserving (TV-like, p-norm) regularization for imaging problems. This is genuinely valuable for
**ill-posed inverse problems** (tomography, deblurring) where a good `damp` is not known a priori -- but
that is not this library's primary workload. Per `docs/features/sparse-bsr.md`'s own benchmark framing,
the library's real target is well-posed-ish sparse BSR systems (structural, sensor-fusion, Jacobian-like
operators) where a caller either already knows a reasonable `damp` or is solving a consistent/
well-conditioned system with `damp=0`. **Verdict: does not change the LSQR/LSMR KEEP verdict; worth
noting as a possible future AUGMENT on top of Section 4's `[A;lambda L]` generalized-Tikhonov item (an
automatic-parameter-selection layer over the same stacked-operator machinery), but explicitly LOWER
priority than every item in Section 4** -- it targets a use case (imaging/inverse problems with unknown
regularization) this library does not currently serve, and adds real complexity (an inner GCV/L-curve
subroutine operating on the small projected bidiagonal system).

**LSLQ / LNLQ reconsidered:** per the task's framing, checking whether these change the verdict --
they do not. LSLQ (Estrin, Orban & Saunders, SIAM J. Matrix Anal. 2019) and LNLQ are a *third* pair
(the SYMMLQ-style siblings of LSQR/CRAIG, as already captured in Section 2/4d), distinguished by an
error-norm bound property, not by dominating LSQR or LSMR on the residual/optimality metrics this
audit is about. They remain GAP/low-priority exactly as scoped in Section 4d; nothing here elevates
them.

**No other post-2010 Golub-Kahan-based LS solver was found that dominates LSQR/LSMR outright** for
this library's deterministic, single-node, sparse-BSR-focused profile -- the field's post-2010 activity
is concentrated in (a) the hybrid/regularization-selection direction just covered, (b) the
error-norm-bound LSLQ/LNLQ direction (Section 4d), and (c) the randomized direction covered next
(Section 7.3), not in a straight algorithmic improvement on the base LSQR/LSMR recurrence itself.

### 7.3 Randomized sketch-and-precondition (Blendenpik, LSRN) -- KEEP-OUT, do not adopt

Blendenpik (Avron, Maymounkov & Toledo, 2010) and LSRN (Meng, Saunders & Mahoney, 2014) build a
right-preconditioner (Section 3's mechanism) from a **random** subspace embedding -- typically a
randomized Walsh-Hadamard or Gaussian sketch of `A` -- then run a handful of preconditioned LSQR
iterations to polish. They target extremely large, well-conditioned, roughly-square-ish **dense** (or
densely-embeddable) systems where forming a full QR/SVD is too expensive but a cheap randomized sketch
gives a near-perfect right-preconditioner in one shot; LSRN additionally targets embarrassingly-parallel
multi-node/GPU settings.

**Two independent disqualifiers for this library, both concrete, not just "flag and move on":**
1. **Determinism.** The sketch is a random projection. Per this project's committed direction
   (`determinism-conformance-harness`/`DetMath` work: deterministic-by-construction core, `FloatMode.Strict`
   for cross-arch reproducibility), any random component must run through a fixed, portable, seeded PRNG
   stream to stay reproducible -- achievable in principle (the library already has deterministic samplers
   elsewhere), but it is not "note the caveat and proceed": a bit-reproducible structured random
   projection needs its own **fast structured transform kernel** (a Walsh-Hadamard or DCT-style
   transform, i.e. new FFT-adjacent machinery beyond what a fixed-seed dense Gaussian sketch would need)
   to be affordable at all -- a Gaussian sketch without a fast transform costs `O(mn)` per sketch, which
   defeats the entire point of the method (it exists specifically to avoid `O(mn)` factorization work).
2. **Workload fit.** This library's real target, per every existing benchmark table
   (`docs/features/sparse-bsr.md`, `docs/features/least-squares.md`), is **sparse BSR**, not huge dense
   systems. A dense random sketch `S*A` of a sparse `A` is generally *dense* (destroys the sparsity that
   makes our `ApplyBlock`/BSR machinery valuable in the first place) unless a sparsity-preserving sketch
   (CountSketch / a sparse Johnson-Lindenstrauss transform) is used instead -- a different, less-studied
   variant of the same idea, with its own correctness/collision-rate analysis to port. Blendenpik/LSRN's
   own motivating use case (huge dense, or GPU/multi-node embarrassingly-parallel) does not match this
   library's single-node deterministic BSR profile at all.

**Recommendation: explicit KEEP-OUT.** Do not adopt Blendenpik/LSRN or any random-sketch preconditioner
for the current roadmap. The cost (a whole new deterministic fast-transform or sparse-sketch subsystem)
is disproportionate to the value (a technique aimed at a workload -- huge dense LS -- this library does
not target), and it directly conflicts with the project's determinism commitment unless that new
subsystem is built first. Revisit only if a future *opt-in, explicitly-nondeterministic, large-dense*
use case is deliberately added as a separate build mode -- not a change to propose speculatively here.

### 7.4 Is our Golub-Kahan bidiagonalization kernel itself current best practice?

**Yes, by design, and the codebase already demonstrates it knows the difference.** There are two
independent Golub-Kahan-family implementations in this library, solving two different problems, and
each uses the numerically-correct choice for its own problem:

- **`lsqr`/`lsmr`'s bidiagonalization** (`Krylov.fProxy.cs` lines 1187-1196 / 1512-1523): plain,
  **no reorthogonalization** of the `u`/`v` vectors. This is not a gap -- it is the deliberate, textbook
  design of LSQR (Paige & Saunders 1982) and LSMR (Fong & Saunders 2010), both of which are celebrated
  specifically for **not needing** reorthogonalization to solve the least-squares/least-norm problem
  accurately: unlike Lanczos-for-eigenvalues (where loss of orthogonality causes spurious "ghost"
  duplicate Ritz values), LSQR/LSMR only use the bidiagonalization to drive a *solution recurrence*, not
  to extract spectral information from the basis itself, and Paige & Saunders' stability analysis shows
  the solve remains as accurate as `A`'s conditioning allows even after orthogonality is lost. Every
  reference implementation surveyed here (SOL's `lsqr.m`/`lsmr.m`, Belos' `LSQRIter`, Krylov.jl) likewise
  ships with no reorthogonalization option for the base solvers. **No change needed; CRAIG/CRAIGMR
  (Section 4a), which reuse this exact loop, inherit the same correct no-reorth design** -- the BIT 1995
  CRAIG paper does not call for reorthogonalization either, for the identical reason.
- **`SVD.LowRank.fProxy.cs`'s truncated GKL** (used for `SVD.svdTruncated`, the `[[svd-truncated-gkl]]`
  feature): **does** implement modern partial reorthogonalization -- a PROPACK-style (Larsen) omega-recurrence
  tracking orthogonality-loss estimates, triggering full DGKS (Daniel-Gragg-Kaufman-Stewart) double
  reorthogonalization only when a semiorthogonality threshold is crossed, plus Extended Local
  Reorthogonalization every step (`SVD.LowRank.fProxy.cs` lines 20-26, 120-241). This is the
  numerically-correct choice for *this* problem, because extracting **singular triplets** from the
  Krylov basis is exactly the eigenvalue-adjacent case where loss of orthogonality causes spurious/ghost
  singular values -- the same failure mode reorthogonalization exists to prevent in Lanczos eigensolvers.

So the two GKB implementations differ *because they should*: one drives a linear solve (no reorth
needed, matches every permissively-licensed reference), the other extracts spectral information (partial
reorth needed, and already implemented at a PROPACK-level of sophistication). **No modern
better-conditioned variant of the plain LSQR/LSMR bidiagonalization is being missed** -- reorthogonalized
GKB would only be relevant here if a future hybrid/regularization-selection method (Section 7.2) needed
a more accurate small-scale projected subspace for its parameter-selection subproblem; that caveat is
noted in the hybrid-methods literature and is worth remembering **if** Section 7.2's hybrid direction is
ever picked up, but it is not a gap in the base LSQR/LSMR/CRAIG solvers audited here.

### 7.5 Summary verdict table

| Solver | Verdict | Rationale | Action |
|---|---|---|---|
| LSQR | **KEEP** | Not dominated by LSMR (Fong-Saunders: "never very far behind"); retains a genuine, literature-documented `\|\|r\|\|`-monotonicity guarantee | None required; optionally re-wire its stopping test to its own monotonic `phibar` quantity (Section 7.1) -- separate follow-up spec, not resolved here |
| LSMR | **KEEP -- preferred default** | Already correctly wired to its own proven-monotonic `\|\|A^T r\|\|`; safer early termination on ill-conditioned `A` | Recommend as the default in future facade/doc work; no code change |
| Hybrid/flexible LSQR (GCV-selected `damp`) | **not adopted** | Targets ill-posed inverse problems (imaging), not this library's sparse-BSR workload; real added complexity | Note as a possible future AUGMENT on top of Section 4's `[A;lambda L]`, explicitly lower priority than every Section 4 item |
| Blendenpik / LSRN (randomized sketch) | **KEEP-OUT** | Conflicts with the project's determinism commitment (random sketch); wrong workload fit (targets huge dense, we target sparse BSR) | Do not adopt; revisit only under a hypothetical opt-in nondeterministic build mode, not proposed here |
| Bidiagonalization kernel (no-reorth in lsqr/lsmr) | **current best practice, KEEP** | Textbook LSQR/LSMR design; matches every permissive reference; CRAIG inherits it correctly | None |
| Bidiagonalization kernel (partial reorth in SVD.LowRank) | **current best practice, KEEP** | Already PROPACK-level (omega-recurrence + DGKS); correct choice for spectral extraction | None; cross-reference only |
