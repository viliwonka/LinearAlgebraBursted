# Spec: LNLQ — least-norm solver with a certified error-in-x bound

## 1. Why (the capability gap)

No solver in the library exposes a **forward error bound** `‖x* − x_k‖` on the solution.
`LstsqInfo` carries `rnorm`/`Arnorm`/`xnorm`; `SolveInfo` carries `rnorm` only. Residual norms
support condition-scaled stopping heuristics but never a certified solution-error bound.

LNLQ (Estrin, Orban & Saunders) fills exactly that gap for the **least-norm** problem
`min ‖x‖ s.t. Ax = b` (consistent, underdetermined `A`, Rows ≤ Cols). It is equivalent to
SYMMLQ on the normal equations of the second kind (`AAᵀy = b, x = Aᵀy`) — the LQ-based sibling of
CRAIG (which is CG on the same equations). Given an **underestimate `est` of σ_min(A)**, LNLQ
produces a monotone, constant-time-per-iteration **upper bound on `‖x* − x_k‖`**. That bound — not
the solution (which equals CRAIG's) — is the reason this method earns its place. Without an `est`,
LNLQ still solves, but adds nothing over CRAIG, so the bound path is the point.

## 2. Reference chain (permissive: papers, not code — stashed in `reference/rectangular/`)

- **`LNLQ-Estrin-Orban-Saunders-eos2018.txt`** — the target method. **Algorithm 2** (core LQ
  recurrence, §3.5, ~line 703) is the solve; **§5** (eqs 38–43, ~lines 958–1050) gives the error
  bounds; **§5.1** (~lines 906–980) gives the `est`-modified factorization `L̃_k` (36).
- **`EOS2017.txt`** — LSLQ companion. §4 gives the Gauss-Radau **node procedure** for choosing
  `ω_k` so σ_min(L̃_k) = `est` (LNLQ §5.1 says its procedure is "identical to Estrin et al. 2017").
- **`EOS2016.txt`** — "Euclidean-norm error bounds for SYMMLQ and CG". **Algorithm 1** (§4,
  ~lines 803–837, "SYMMLQ with CG error estimation") is the **constant-time error-bound recurrence**
  from `{α_k, β_k}`; §4's recurrences (~line 487) define the ξ eigenvector-element recurrence the
  node procedure needs. This is the self-contained source for the bound machinery.

The MPL-2.0 Julia `Krylov.jl/lnlq.jl` may be consulted ONLY as a numeric oracle in a gitignored
scratch (never copied into the tree). All ported logic derives from the three papers above.

## 3. Structural sibling to mirror

`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.CRAIG.fProxy.cs` — copy its shape wholesale:
- matrix-free over `IfProxyLinearOperator`; reuse `GolubKahanVStep`/`GolubKahanUStep` (identical
  Golub-Kahan process, Algorithm 1);
- the same 8-overload ladder: generic `TOp`; dense primitive + arena + default; BSR primitive + BSR
  with caller `AT` + BSR arena (materializes `AT`) + BSR default;
- no warm start (min-norm characterization requires `x₀ = 0` — zero `x` internally);
- `b = 0 ⟹ Converged, x = 0` early-out;
- verify-at-exit: reconcile a claimed `Converged` against a certified-exact residual before
  reporting (reuse `lstsqResidual`; CRAIG's `CraigInfo` pattern).

New file: `Krylov.LNLQ.fProxy.cs` (template), generated → `Source/…`. Register nothing by hand.

## 4. Public API

```
LnlqInfo lnlq<TOp>(in TOp A, in fProxyN b, ref fProxyN x,
                   ref fProxyN u, ref fProxyN v,
                   ref fProxyN w, ref fProxyN wbar,        // LQ direction pair (skippable — see below)
                   ref fProxyN tmpM, ref fProxyN tmpN,
                   int maxIter, fProxy tol, fProxy sigmaMinEst)
    where TOp : struct, IfProxyLinearOperator
```

- `sigmaMinEst` = underestimate of σ_min(A). Semantics: `sigmaMinEst <= 0` (or NaN) ⟹ run the solve
  with **no** bound; `LnlqInfo.xErrBound = fProxy.NaN`. `sigmaMinEst > 0` ⟹ compute the bound. The
  bound is a valid **upper** bound only when `sigmaMinEst <= σ_min(A)`; if the caller supplies too
  large a value the bound may under-estimate — document this, do NOT clamp or assert it.
- Arena/default overloads default `sigmaMinEst` to 0 (bound off) so the cheap call shape stays cheap.
- Provide a `(… ) with sigmaMinEst` explicit overload and the bound-off convenience overloads.

### LnlqInfo (dedicated struct — do NOT overload `LstsqInfo`)

`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SolveInfo.cs`, mirror `LstsqInfo`'s shape:
- `double rnorm` (‖b − Ax‖, certified), `double xnorm` (‖x‖),
- `double xErrBound` — upper bound on `‖x* − x‖` for the RETURNED iterate; `NaN` when no `est`,
- `int iterations`, `IterativeSolveStatus status`,
- `bool Solved => status == Converged`, implicit `bool`, `ToFixedString()`/`ToString()`.

## 5. Which iterate to return (spec decision — flag for review)

Algorithm 2 maintains both the LNLQ point `x_k^L` (eq 43 bound) and the CRAIG-transfer point
`x_k^C` (eq 41 bound, always the tighter/more-accurate iterate). **Return `x_k^C`** (the transfer
point — same solution CRAIG converges to) and report `xErrBound` from **eq (41)**
`‖x* − x_k^C‖² ≤ ρ̃_k² − ρ_k²`. Rationale: it is the more accurate iterate and its bound is
tighter, so the returned `x` + `xErrBound` form the strongest honest pair. Keep the `x_k^L` /
eq (43) machinery only if it is needed to compute the transfer (per Algorithm 2 it is). If returning
`x_k^L` turns out cleaner/more faithful to the reference, that is an acceptable alternative — note
the choice in the DEVLOG either way.

Per EOS2018 §3.5 (lines 277–281): to get only `x`, the `w`/`wbar`/`yL`/`yC` vectors are NOT needed
unless recovering `x = Aᵀy`. Prefer the `x`-only path (skip the `y` machinery) to keep the buffer
count near CRAIG's; drop `w`/`wbar` params if the chosen formulation doesn't use them.

## 6. Error-bound recurrence (the hard, must-be-faithful part)

Implement the constant-time bound from **EOS2016 Algorithm 1** adapted to the Golub-Kahan setting
per **EOS2018 §5 / §5.1** and the **EOS2017 §4** node procedure:
1. Choose `ω_k` (the est-Radau node) so σ_min(L̃_k) = `est` — EOS2018 line 343:
   `ω_k = √(est² − est·ξ²_{k−2})`, with the `ξ` eigenvector-element recurrence from EOS2016 §4.
2. Maintain the tilde factors `t̃_k = (t_{k−1}, τ̃_k)`, `z̃_k = (z_{k−1}, ζ̃_k)` (EOS2018 lines
   975–977) with `ρ̃_k = α̃_k s_k`, etc.
3. `xErrBound = √(eq 41)` for the returned `x_k^C`.

**Do NOT hand-derive any of this.** Port it line-by-line from the papers' recurrences. If a recurrence
element is ambiguous in the OCR'd `.txt`, read the corresponding `.pdf` page (poppler is installed;
`pdftotext -layout` or open the PDF) rather than guessing. The **sliding-window** refinement (EOS2018
eq 40, tightens the *y* bound) is OUT OF SCOPE for v1 — note it as a future refinement in the DEVLOG.

## 7. Tests (oracle-based; prove red pre-fix)

`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LNLQTests.fProxy.cs`, templated float+double,
run THROUGH an IJob where the sibling CRAIG/lstsq tests do. Oracles:
1. **Solution correctness** — for a full-row-rank underdetermined `A`, `x == LQ.minNormSolve(A, b)`
   to tol, AND `x ≠ x_true_of_a_wrong_system` guard (don't assert a tautology). Also `x == craig(A,b)`
   to tol (they share the min-norm point) — a cross-solver consistency oracle.
2. **Bound is a true upper bound (the headline invariant)** — with `est = (1 − 1e-10)·σ_min(A)`
   (a valid underestimate; compute σ_min via the library SVD/`SVD.values` on a small dense case),
   assert `xErrBound >= ‖x_k − x*‖` at the returned iterate — and, if cheap to instrument in the
   test, at every iteration (monotone, never violated). This is the guard-shaped invariant that
   actually certifies the feature. Use a small, well-conditioned `A` so σ_min is trustworthy.
3. **Bound tightness (sanity, not a hard gate)** — with a good `est`, `xErrBound` should be within a
   small factor (say ≤ 10×) of the true error near convergence. Assert loosely; the paper shows
   tightness but finite precision can loosen it — do not over-constrain.
4. **No-est path** — `sigmaMinEst <= 0 ⟹ xErrBound is NaN`, solve still correct.
5. **s=1 / square** — on a square nonsingular `A`, recovers the unique solution; bound still valid.
6. **Zero-b ⟹ Converged, x=0, xErrBound = 0**; **rank-deficient ⟹ Breakdown** (mirror CRAIG's
   breakdown contract; do not fabricate a bound on a breakdown exit).

Before/after discipline: a deliberately-wrong bound (e.g. returning `rnorm/est` instead of the
Gauss-Radau bound) must FAIL test 3's tightness and pass test 2 — confirm the tightness test has
teeth by checking it rejects the crude `rnorm/est` bound during development.

## 8. Acceptance

- All LNLQTests green (float+double); full suite `Result=Passed failed=0`.
- `xErrBound` never violates the true error in test 2 across the iteration history.
- README/CHANGELOG: one honest line each (user hand-writes prose — provide a facts-only skeleton).
- DEVLOG entry under `## Krylov.LNLQ`: method, the 3-paper reference chain, the returned-iterate
  choice (§5), sliding-window deferred, and the `est`-validity caveat.

## 9. Out of scope (v1)

- Regularization (EOS2018 §4) and the Generalized-Golub-Kahan preconditioned path (§6) — defer.
- Sliding-window y-bound refinement (eq 40) — defer.
- The `y` solution / `x = Aᵀy` recovery path — implement only if it falls out of the chosen
  formulation for free; otherwise defer.

## 10. Build/test commands (headless, FOREGROUND)

`pwsh` / PowerShell 7 is NOT installed. Use Windows PowerShell 5.1, run FOREGROUND (backgrounding a
Unity run orphans the process):
- regen: `powershell.exe -NoProfile -File Tools/regen.ps1`
- tests: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File Tools/run-tests.ps1`

Edit TEMPLATES only (never `Source/*`). Contracts-only comments; rationale/history → DEVLOG.
