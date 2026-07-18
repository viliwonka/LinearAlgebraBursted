# Spec: Chebyshev polynomial preconditioner over BSR

Status: DRAFT (implementation spec for the coder agent)
Scope: one new preconditioner struct + arena factory + pcg/pminres convenience rungs + tests.
Related: `docs/dev/spec-multigrid-solver.md` (this preconditioner IS the planned AMG smoother —
see §8), `docs/dev/spec-sparse-bsm.md`.

## 1. What and why

`fProxyChebyshev` implements `IfProxyPreconditioner`: `Apply(in r, ref z)` produces `z ≈ A⁻¹ r`
for a square SPD BSR `A` by running a degree-`degree` Chebyshev three-term recurrence on the
point-Jacobi-scaled operator `D⁻¹A` over an interval `[lo, hi]` bracketing its upper spectrum.

The Apply is pure spMV + elementwise diagonal scale + AXPY:

- zero triangular solves — unlike `fProxyIC0`/`fProxySSOR`, whose forward/backward sweeps are
  inherently sequential, every kernel in the Chebyshev Apply is an already-SIMD-optimized
  streaming pass (`BSR.spMV`, axpy);
- **dot-free** — the recurrence needs no reductions at all, so Apply is deterministic by
  construction (only `+ − × /` in fixed order; the single `sqrt` lives in setup);
- doubles as the AMG smoother later (§8) with zero rework.

Honest expectation (do not oversell in docs): CG is optimal over Krylov spaces, so
Chebyshev-PCG does not beat Jacobi-scaled plain CG in *total spMV count* — its value is (a) far
fewer outer iterations ⇒ far fewer dot-product/convergence-check passes per spMV, (b) the AMG
smoother role, (c) a fully-SIMD deterministic rung where IC0/SSOR sweeps are sequential. Tests
assert on outer-iteration reduction, not wall clock.

## 2. Survey facts (anchors, verified 2026-07-18)

- `IfProxyPreconditioner` — `TemplateSource/Interfaces/LinearOperator.fProxy.cs:54`. Single
  member `void Apply(in fProxyN r, ref fProxyN z)`; no Rows/Cols on the interface
  (implementations validate against their own stored shape).
- Two existing preconditioner shapes:
  - record-table + IDisposable: `fProxyBlockJacobi` (`Sparse/fProxyBlockJacobi.cs:22`,
    `fProxyBlockJacobiRecord` in `fProxyBSRRecords.fProxy.cs:47`);
  - **composed-of-arena-pieces, no record table, no Dispose**: `fProxySSOR`
    (`Sparse/fProxySSOR.cs:32`) and `fProxyIC0` (`Sparse/fProxyIC0.cs:25`). SSOR owns two
    arena scratch vectors (`Scratch1/Scratch2`) as readonly fields, set once in the ctor.
    **Chebyshev follows this second shape.**
- Arena factories — `Sparse/Arena.Sparse.fProxy.cs`: `fProxyBlockJacobi(in A[, out
  PreconditionerInfo])` (:78/:103), `fProxySSOR(in A[, fProxy omega])` (:216/:223),
  `fProxyIC0(in A[, out info])` (:235/:243), `fProxyILU0(...)` (:255/:263). SSOR's factory is
  the model: an Arena instance method that null-checks `_core`, wraps in
  `EnterMutation/ExitMutation`, and calls the `(in A, ..., ref Arena)` ctor.
- Convenience-overload ladder — each of BlockJacobi/SSOR/IC0 has exactly **three rungs** on
  `Krylov.pcg` (`OP/Krylov.fProxy.cs:373–473`) and three on `Krylov.pminres`
  (`OP/Krylov.PMinres.fProxy.cs:230–342`): (1) zero-alloc with caller scratch `ref r, ref p,
  ref Ap, ref z, maxIter, tol`; (2) allocating via `b.fProxyTempVec(A.M_Rows)`; (3) defaults
  `maxIter = A.M_Rows`, `tol = Consts.fProxySqrtEps`. Each forwards through
  `new fProxyBSROperator(in A)` into the generic `pcg<TOp,TPre>`/`pminres<TOp,TPre>` core.
  **`pbiCGStab` has convenience rungs only for `fProxyILU0`** (`OP/Krylov.PBiCGStab.fProxy.cs:115/:132`)
  — BlockJacobi/SSOR/IC0 have none there, so "mirror exactly" = pcg + pminres only (§10 Q3).
- Kernels to build from (do NOT hand-roll):
  - `BSR.spMV(in fProxyBSR, in fProxyN, ref fProxyN)` — `Sparse/SparseOP.fProxy.cs:21`;
    **handles Symmetric (lower-block-only) storage internally** (branch at :42), so Chebyshev
    needs no mirror-to-full (unlike SSOR, whose sweeps forbid Symmetric storage).
  - `Blas.scaledCopy(a, in x, ref y)` (`OP/Blas.Fused.fProxy.cs:53`), `Blas.dot`,
    `Blas.updateXR`; extensions `y.addScaledInPlace(a, x)` (y += a·x) and
    `y.scaleAddInPlace(a, x)` (y = a·y + x) in `OP/OP.Component.fProxy.cs:85/:94`;
    raw `UnsafeOP.axpy`/`scaledCopy` with `[NoAlias]` (`OP/UnsafeOP.fProxy.cs:1746/:1980`).
- Eigenvalue estimation, both deterministic (fixed seed `v[i] = 1 + (i & 3)`, fixed-order
  reductions):
  - `Eigen.lanczos<TOp>(in A, ref fProxyLanczosCache ws, ref eigenvalues, steps[, breakdownTol])`
    — `OP/Eigen.fProxy.cs:678`; twice-reorthogonalized, fixed `steps`, workspace via
    `arena.fProxyLanczosCache(n, steps)` (`OP/Eigen.LanczosWorkspace.fProxy.cs:78`). Ritz
    values via the tridiagonal T; extremal values converge first (Kaniel–Paige–Saad) — ideal
    for λ_max in ~10 steps.
  - `Eigen.powerIteration<TOp>` — `OP/Eigen.fProxy.cs:38` (fallback option, slower to converge).
- Scaled-operator precedent: `fProxyColScaledOperator<TInner>`
  (`Interfaces/LinearOperator.fProxy.cs:158`) — a readonly wrapper owning a `Scratch` vector.
  The symmetric Jacobi-scaled wrapper in §4.3 copies this pattern.
- `PreconditionerInfo` — `Sparse/PreconditionerInfo.cs:16` (`status`, `shift`, `attempts`).
- Test bed: `TemplateSourceTests/fProxy/SparseSolverTests.fProxy.cs`
  (class `fProxySparseSolverTests`; existing pcg/preconditioner tests, aliasing-throw tests,
  the non-SPD-preconditioner breakdown test at :507/:1707). Poisson generator:
  `arena.fProxyLaplacian2D(gridX, gridY)` (`Sparse/Gallery.Sparse.fProxy.cs:182`, BSR
  block-tridiagonal 2D stencil; `PCGBenchmark.fProxy.cs:129` documents its shape).
- Warm-state rule (job-struct-copy audit; LOBPCG IJob cache-copy bug): a solver/preconditioner
  struct passed into an IJob is a value copy — writes to its fields are lost, and pointers to
  its own fields go stale. Therefore: **no mutable fields, no post-construction writes**. All
  Chebyshev state (interval, diagonal, scratch) is readonly, set once at construction,
  arena-buffer-backed — the same reason `fProxySSOR`/`fProxyIC0` are IJob-safe by construction.
- Comment policy: contracts only in code; history/benchmarks go to `Sparse/DEVLOG.md`.

## 3. Algorithm

### 3.1 Operator and interval

Default and only v1 form: **point-Jacobi scaling**. `D = diag(A)` (scalar diagonal entries;
SPD ⇒ all positive — throw otherwise). The preconditioner realizes

```
z = q(D⁻¹A) · D⁻¹ r,     q = degree-`degree` Chebyshev polynomial on [lo, hi]
```

Setup computes:

- `InvDiag[i] = 1 / A[i,i]` — one pass over the stored diagonal blocks (both storage modes
  store the diagonal block; scan each block-row for `ColInd == i`, the same scan
  `fProxyBlockJacobi`'s ctor uses).
- `hi = safety · λ̃max(D⁻¹A)`, with `λ̃max` = largest Ritz value from `Eigen.lanczos<TOp>` run
  for a **pinned** `eigSteps` steps (no early stopping other than lanczos's own deterministic
  breakdown path) on the **symmetrically scaled** operator `S·A·S`, `S = D^(-1/2)` — same
  spectrum as `D⁻¹A` but symmetric, so lanczos applies (§4.3 wrapper). `safety` default 1.1.
- `lo = hi / kappa`, `kappa` default 30.

Interval semantics (state in the struct doc): this is the *preconditioner* form, not full
inversion — the polynomial strongly damps components with eigenvalues in `[lo, hi]` (the upper
spectrum) and leaves the few components below `lo` to the outer Krylov iteration. Larger
`kappa` reaches further down the spectrum with a weaker per-mode damping.

SPD contract (this is the real safety story — document it): the induced `M⁻¹ = q(D⁻¹A)D⁻¹` is
symmetric wrt the standard inner product and positive definite **as long as no eigenvalue of
`D⁻¹A` exceeds `hi`** (`1 − p(t) > 0` for all `t ∈ (0, hi]`, where `p` is the shifted-scaled
Chebyshev error polynomial; below `lo` is harmless, above `hi` `p` can exceed 1 and flip the
sign). Underestimating λ_max is the failure mode; overestimating merely weakens the
preconditioner. That is what the 1.1 safety factor is for; `pcg`'s existing NaN-safe
`⟨r,z⟩ > 0` breakdown guard (`Krylov.fProxy.cs:296`) is the runtime backstop.

### 3.2 Recurrence (exact form, fixed coefficient order)

Zero-initial-guess Chebyshev iteration (Saad, Iterative Methods 2nd ed., ch. 12.1, folded to
the preconditioner-apply form). All scalar coefficient arithmetic in exactly this order:

```
theta   = (hi + lo) / 2
delta   = (hi - lo) / 2
sigma   = theta / delta
rhoPrev = 1 / sigma          // = rho_0
invTheta = 1 / theta
```

Apply(r → z), with owned scratch `d`, `rk`, `t` (length n):

```
d  = invTheta * (InvDiag ∘ r)          // d_0 ; elementwise scale
z  = d                                  // z_1
rk = r                                  // running residual r_1 pre-update = r
repeat degree times:                    // k = 1 .. degree
    t   = A d                           // BSR.spMV — the one spMV per step
    rk -= t                             // rk = r_k  (addScaledInPlace(-1, t))
    rho = 1 / (2*sigma - rhoPrev)
    a   = rho * rhoPrev
    b   = 2 * rho / delta
    d   = a*d + b*(InvDiag ∘ rk)        // three-term update
    z  += d
    rhoPrev = rho
```

Output `z = z_{degree+1}`, a polynomial of degree `degree` in `D⁻¹A` applied to `D⁻¹r`.

- **Cost per Apply: exactly `degree` spMVs**, `degree+1` diagonal-scale passes, ~2·`degree`
  axpys. `degree = 0` would degenerate to scaled point-Jacobi `z = (1/θ)D⁻¹r`; require
  `degree >= 1`, default **3** (sane range 2–4; §10 Q1).
- The `d`-update is one fused elementwise pass: `d[i] = a*d[i] + b*InvDiag[i]*rk[i]`; the
  `z += d` can fold into the same pass. Implement first by composing existing kernels
  (`addScaledInPlace` + a small loop); add a single fused `[NoAlias]` `UnsafeOP` kernel only
  if the kernel benchmark justifies it (A/B rule — never drop/add unsafe fusion without a
  benchmark). Keep the loop raw-pointer-hoisted per the hoist-pass idiom either way.
- `r` is `in` and never written (Apply copies into `rk` first). `z` must not alias `r` or any
  scratch — same up-front pointer guard as `fProxySSOR.Apply`.

## 4. API

### 4.1 Options

```csharp
// TemplateSource/Sparse/fProxyChebyshev.cs
public struct fProxyChebyshevOptions
{
    public int degree;      // spMVs per Apply; >= 1. Default 3.
    public fProxy kappa;    // lo = hi / kappa; > 1. Default 30.
    public int eigSteps;    // pinned Lanczos steps for hi; >= 1. Default 10.
    public fProxy safety;   // hi = safety * lambdaMaxEstimate; >= 1. Default 1.1.
    public static fProxyChebyshevOptions Default => ...;
}
```

Proxy-typed fields ⇒ no C# default parameter values anywhere (CS1750 template rule); the
default flows through `Default` + forwarding overloads. Short names per the settled
short-param-names ruling.

### 4.2 Struct

```csharp
public readonly struct fProxyChebyshev : IfProxyPreconditioner
{
    public readonly fProxyBSR A;        // full OR Symmetric storage (spMV handles both)
    public readonly fProxyN InvDiag;    // 1/diag(A), length n
    public readonly fProxy Lo, Hi;      // final interval (diagnostics; already-computed numbers)
    public readonly fProxy Theta, Delta, Sigma;  // precomputed from Lo/Hi
    public readonly int Degree;
    public readonly fProxyN Scratch1, Scratch2, Scratch3;  // d, rk, t — arena-owned
    public int Rows => A.M_Rows;

    public fProxyChebyshev(in fProxyBSR a, in fProxyChebyshevOptions opt, ref Arena arena);
    public fProxyChebyshev(in fProxyBSR a, ref Arena arena);   // Default options
    public void Apply(in fProxyN r, ref fProxyN z);
}
```

- Composed-of-arena-pieces like SSOR/IC0: no record table, no `Dispose()` — the arena owns
  every buffer. All fields readonly, set once ⇒ IJob-safe by construction (§2 warm-state rule).
- Ctor throws `ArgumentException` on: non-square (`BlockRows != BlockCols || BR != BC`);
  missing diagonal block; any `A[i,i] <= 0` (message: "... is A symmetric positive
  definite?"); `degree < 1`; `kappa <= 1`; `eigSteps < 1`; `safety < 1`.
- Apply throws on size mismatch and on `z` aliasing `r` (pointer compare, SSOR idiom).
  Owned scratch ⇒ one Apply at a time per struct instance (same implicit contract as SSOR).
- No `out PreconditionerInfo` overload in v1: setup has no shift-retry/factorization-failure
  path — every failure is a caller-contract violation and throws (§10 Q6).

### 4.3 Internal scaled operator (setup only)

```csharp
// internal, same file; pattern copied from fProxyColScaledOperator
internal readonly struct fProxyJacobiScaledBSROperator : IfProxyLinearOperator
{
    public readonly fProxyBSR A;
    public readonly fProxyN InvSqrtD;   // D^(-1/2), length n — sqrt is in the deterministic set
    public readonly fProxyN Scratch;    // length n
    // Apply: s = InvSqrtD ∘ x ; t = A s (spMV into y) ; y = InvSqrtD ∘ y
    // ApplyT == Apply (symmetric); ApplyDot composes Apply + Blas.dot;
    // ApplyBlock: per-row via Apply (ColScaledOperator's Temp-buffer fallback) — lanczos
    // never calls it, present only to satisfy the interface.
}
```

Setup then runs `Eigen.lanczos(in scaledOp, ref ws, ref ritz, eigSteps)` with
`ws = arena.fProxyLanczosCache(n, eigSteps)` and takes the max over the first `info.produced`
Ritz values. `InvSqrtD` and the wrapper's scratch are arena vectors freed with the arena
(setup-only garbage; acceptable, matches how lanczos's own cache is handled).

### 4.4 Arena factory

In `Sparse/Arena.Sparse.fProxy.cs`, next to the SSOR factories (:216):

```csharp
public fProxyChebyshev fProxyChebyshev(in fProxyBSR A, in fProxyChebyshevOptions opt); // ctor(ref this)
public fProxyChebyshev fProxyChebyshev(in fProxyBSR A);                                 // Default
```

Same `_core` null-check + `EnterMutation/ExitMutation` bracketing as `fProxySSOR`'s factory.

### 4.5 Solver convenience rungs

Mirror the BlockJacobi/SSOR/IC0 rungs **exactly** — three each, forwarding through
`new fProxyBSROperator(in A)`:

- `Krylov.pcg(in fProxyBSR A, in fProxyChebyshev M, ...)` × 3, placed after the IC0 rungs in
  `OP/Krylov.fProxy.cs` (:445–473 block is the copy model).
- `Krylov.pminres(in fProxyBSR A, in fProxyChebyshev M, ...)` × 3 in
  `OP/Krylov.PMinres.fProxy.cs` (:312–342 model).
- **No pbiCGStab rung** (matches BlockJacobi/SSOR/IC0, which have none; Chebyshev's SPD
  interval assumption belongs with the SPD solvers — §10 Q3).
- No LOBPCG overloads in v1 (BlockJacobi has them; §10 Q7).

## 5. Cost model (document in the struct doc, verify in the benchmark)

- Setup: one diagonal pass O(nnzb-diag·BR) + `eigSteps` spMVs + O(eigSteps²·n) reorth — one
  time, roughly `eigSteps` PCG iterations' worth of work.
- Apply: `degree` spMVs ⇒ a Chebyshev-PCG outer iteration costs `degree + 1` spMVs (one in
  pcg itself). Break-even vs Jacobi-scaled cg needs outer iterations to shrink by more than
  `degree + 1`× — approximately what the Chebyshev bound delivers *inside* `[lo, hi]`; the
  net effect is fewer reductions/norm checks per spMV, not fewer spMVs (§1 honesty note).
- Extend `TemplateSourceBenchmarks/PCGBenchmark.fProxy.cs` (`BenchPrecondCoreFProxy`, :138,
  already runs Laplacian2D + random-SPD across preconditioners) with a Chebyshev column.

## 6. Determinism

- **Apply**: dot-free; `BSR.spMV`, axpy, elementwise scales are fixed-order streaming kernels;
  coefficients are scalars computed in the pinned order of §3.2. Operations used: `+ − × /`
  only ⇒ inside the cross-arch-deterministic set (determinism-analysis rules); no
  transcendentals, no RNG, no data-dependent iteration count (`degree` is fixed).
- **Setup**: deterministic function of the input bits — lanczos seeds deterministically
  (`v[i] = 1 + (i & 3)`), runs pinned `eigSteps`, and its dot reductions use the fixed-tree
  SIMD kernels (same-arch bit-reproducible; the sum/dot reorder waiver noted in `OP/DEVLOG.md`
  applies, as it already does to every pcg solve). `sqrt` (for `InvSqrtD` and inside lanczos)
  is in the allowed set. Net: same input ⇒ bit-identical `hi/lo` ⇒ bit-identical Apply.

## 7. Symmetric-storage note

`BSR.spMV` natively handles Symmetric (lower-block-only) storage, and the diagonal blocks are
stored in both modes — so `fProxyChebyshev` accepts either storage with **no mirror-to-full**
(a concrete advantage over SSOR, which pays `fProxyBSRMirrorToFull`). Say so in the struct doc.

## 8. AMG smoother reuse (forward contract)

`docs/dev/spec-multigrid-solver.md` picks "Chebyshev(-Jacobi)" as the smoother and notes it is
"also a valid pcg TPre" (:45). To make reuse mechanical:

- Put the §3.2 recurrence body in a static zero-alloc kernel (suggested home: `BSR` class),
  taking explicit scratch — e.g.
  `BSR.chebyApply(in fProxyBSR A, in fProxyN invDiag, fProxy theta, fProxy delta, fProxy sigma, int degree, in fProxyN r, ref fProxyN z, ref fProxyN d, ref fProxyN rk, ref fProxyN t)`
  — with the zero-initial-guess contract. `fProxyChebyshev.Apply` is a thin forwarder passing
  its own scratch.
- AMG's smoother step with nonzero guess is then `r = b − A x` (one spMV) + `chebyApply` +
  `x += z`, all outside the kernel — no nonzero-guess variant needed in v1.
- Keep the kernel's parameter list free of the struct type so AMG levels (own diagonals, own
  intervals, own buffers) can call it directly.

## 9. Tests (`TemplateSourceTests/fProxy/SparseSolverTests.fProxy.cs`, templated class)

1. **Poisson iteration-count reduction**: `A = arena.fProxyLaplacian2D(g, g)` (g ≈ 32),
   deterministic `b`. Assert all converge at the same tol and
   `iters(pcg, Chebyshev d=3) < iters(pcg, fProxyBlockJacobi) < iters(cg)`.
   (pcg's convergence test is the true residual, same criterion as cg — counts are directly
   comparable, per the pcg doc comment.)
2. **Degree sweep sanity**: d ∈ {1, 2, 3, 4} on the same system — all converge; outer
   iterations non-increasing in d (allow ±1 slack).
3. **Solution correctness**: pcg+Chebyshev solution matches the dense direct solve of the same
   system within tol (existing cross-check pattern in the file).
4. **SPD spot check**: for a few deterministic vectors u, v:
   `dot(u, M.Apply(v)) ≈ dot(v, M.Apply(u))` and `dot(v, M.Apply(v)) > 0`.
5. **Through-IJob**: run the full build-solve (or at least Apply + pcg) inside a Burst IJob via
   `.Run()` and assert bit-identical x vs the main-thread run — the LOBPCG-lesson test shape
   (struct-copy safety is a claim; test it).
6. **Determinism**: two identical solves produce bit-identical x (byte compare or the
   conformance harness's xxHash32 idiom).
7. **Contracts**: non-SPD diagonal (a zero/negative diagonal entry) throws; `degree = 0`,
   `kappa <= 1` throw; `Apply` aliasing z==r throws; size mismatch throws.
8. **Storage-mode equivalence**: Symmetric-storage A and its mirrored full-storage twin give
   bit-identical Apply output.

Burst test gotchas apply (no enum Assert.AreEqual — `IsTrue(a == b)`; CompileSynchronously).

## 10. Open questions

1. **Default degree / kappa** (3 / 30 here, PETSc-GAMG-flavored): tune once on the
   PCGBenchmark Laplacian2D + random-SPD rows before freezing; record verdict in DEVLOG.
2. **eigSteps default (10) and the hi guarantee**: lanczos λ_max converges fast but is a lower
   estimate; `safety = 1.1` is the standard patch, not a proof. Option: also compute the
   one-pass Gershgorin/row-sum bound `max_i Σ_j |(D⁻¹A)_ij|` (guaranteed ≥ λ_max; ~20-line
   BSR pass, nothing shipped today) and use it as a cap/validation or fallback. Decide:
   (a) `safety·lanczos` only (spec'd default), or (b) `min(safety·lanczos, gershgorin)` with
   the caveat that only (pure) gershgorin is guaranteed-safe.
3. **pbiCGStab rung**: none of BlockJacobi/SSOR/IC0 have one (only ILU0). Add a Chebyshev rung
   there anyway, or keep the SPD preconditioners off the nonsymmetric solver? (Also touches
   the open Krylov-preconditioner-coverage TODO for pbiCGStab/pminres generics.)
4. **Un-scaled variant** (D = I, interval on A itself): expose or not? Costless to add as an
   options flag later; v1 ships Jacobi-scaled only.
5. **Block-diagonal scaling**: replace the point diagonal with `fProxyBlockJacobi.DInv`
   (block-Jacobi-Chebyshev — better for BR>1 elasticity blocks; Apply's scale becomes the
   existing `blockJacobiApplyB{b}` kernels, but the Lanczos-side symmetric split needs block
   Cholesky factors). Deferred; the §8 kernel signature (invDiag as a plain vector) would need
   a block twin.
6. **`out PreconditionerInfo` overload**: v1 throws on every failure (no retry path exists).
   Add a non-throwing twin for API uniformity with BlockJacobi/IC0?
7. **LOBPCG overloads** with Chebyshev as `TPre` (BlockJacobi has six): needs an
   interval-vs-smallest-eigenpair story first; deferred.

## 11. File layout / process

- `TemplateSource/Sparse/fProxyChebyshev.cs` — options struct, preconditioner struct, internal
  scaled operator, (kernel if placed here rather than in `SparseOP.fProxy.cs`). No `.fProxy.cs`
  suffix needed (BlockJacobi/SSOR/IC0 precedent; codegen reads all `*.cs`).
- `TemplateSource/Sparse/Arena.Sparse.fProxy.cs` — two factory overloads.
- `TemplateSource/OP/Krylov.fProxy.cs`, `OP/Krylov.PMinres.fProxy.cs` — 3 + 3 rungs.
- `TemplateSourceTests/fProxy/SparseSolverTests.fProxy.cs` — §9 tests.
- `TemplateSourceBenchmarks/PCGBenchmark.fProxy.cs` — Chebyshev column (§5).
- `Sparse/DEVLOG.md` — tuning verdicts, rejected alternatives; **no history in code comments**.
- Regenerate via `Tools/regen.ps1`; run the suite headless via `Tools/*.ps1`. Templates are
  the only files touched — never the generated `Assets/LinearAlgebra/Source` output.
