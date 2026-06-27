# Spec — Literature / test-matrix gallery

Status: **SPEC (coder-ready)** · 2026-06-27 · fProxy-only (float/double)

A curated collection of "famous" matrices with **known closed-form properties** (eigenvalues,
determinant, condition number, inverse, definiteness), as first-class library generators — usable both
by library users and as inputs to the solver/eigen/SVD/Cholesky unit tests (replacing today's inline
construction in `LiteratureTests`). Sources: MATLAB `gallery` (Higham, *Test Matrix Toolbox*),
MatrixDepot.jl, Higham's test-matrix collection.

## Placement — "accessible but not in the obvious place"
- New namespace **`LinearAlgebra.Gallery`**, partial static class **`fProxyGallery`** with **Arena
  extension methods** `arena.fProxy<Name>(…) → fProxyMxN`. Because the namespace is separate, they do
  **not** pollute `arena.` autocomplete for normal `using LinearAlgebra;` code — a caller opts in with
  `using LinearAlgebra.Gallery;`. Tucked away, fully accessible, fluent. (MATLAB's `gallery(...)` model.)
- Method names keep the `fProxy` token (→ `floatPascal`/`doublePascal`): required because two arena
  extensions `Pascal(this ref Arena)` returning `floatMxN` vs `doubleMxN` would be a return-type-overload
  collision. Same reason the existing `fProxyHilbertMatrix` is type-prefixed.
- Split across two files (one partial class) to keep the two build batches conflict-free:
  `Arena/Gallery.SPD.fProxy.cs` and `Arena/Gallery.Special.fProxy.cs`.
- The legacy `arena.fProxyHilbertMatrix` (main namespace) is left untouched for back-compat; the gallery
  adds its own `fProxyHilbert`. (Legacy can be deprecated later.)
- fProxy-only for v1 (float+double). Integer-valued matrices (Pascal, MinIJ, magic, …) still emit
  fProxy entries — exact for the small sizes used. Int variants deferred.

## Conventions
- Allocating, arena-backed (like the existing `fProxyHilbertMatrix` / generators): each returns a fresh
  `fProxyMxN` (persistent arena alloc). `(fProxy)` casts, `ArgumentException("fProxyName: msg")` on bad args.
- 0-based indices in code; formulas below give the 0-based entry. Where a matrix is parametrized
  (nodes / α / ρ / ε), it takes those args.
- Each generator's XML doc states the **known property** (so it's discoverable and self-documenting).

---

## Batch A — SPD / symmetric family  (`Gallery.SPD.fProxy.cs`)
Targets: Cholesky, `eigenDecomposition` (Jacobi), CG, `cond`, `LU.determinant`.

| Generator | Entry (0-based) | Known property (to assert) |
|---|---|---|
| `fProxyHilbert(n)` | `1/(i+j+1)` | SPD, totally positive, ill-conditioned (cond(H₃)≈524.06) |
| `fProxyPascal(n)` | `C(i+j, i)` (binomial) | symmetric, **det=1**, SPD, integer; eigenvalues in reciprocal pairs |
| `fProxyLehmer(n)` | `min(i,j)+1)/(max(i,j)+1)` | SPD, totally nonneg, **cond < 4n²**, tridiagonal inverse |
| `fProxyMinIJ(n)` | `min(i,j)+1` | SPD, **det=1**, inverse = tridiag(−1,2,−1) w/ last diag 1 |
| `fProxyKMS(n, ρ)` | `ρ^|i−j|` | SPD for \|ρ\|<1, **det=(1−ρ²)^{n−1}**, tridiagonal inverse |
| `fProxyPei(n, α)` | `α + (i==j?1:0)`  → i.e. `αI + ones` | eigenvalues **α+n** (×1), **α** (×n−1); det=αⁿ⁻¹(α+n); SPD if α>0 |
| `fProxyMoler(n, α=−1)` | `Uᵀ U`, `U`=upper-tri(1 diag, α above) | SPD, **det=1**, one tiny eigenvalue (build via the triw factor) |
| `fProxyLaplacian1D(n)` | tridiag(−1, **2**, −1) (Strang 2nd-difference) | SPD; **eig λ_k=2−2cos(kπ/(n+1))**; **det=n+1**; cond=λmax/λmin |

(Pei's standard form `αI+ones`: diagonal `α+1`, off-diagonal `1` — write it that way.)

## Batch B — eigenvalue / nonsymmetric / structured / rank  (`Gallery.Special.fProxy.cs`)
Targets: `eigenDecomposition`, `eigenvaluesQR`, SVD, QR/QRCP, least-squares, FFT cross-check, det.

| Generator | Definition | Known property (to assert) |
|---|---|---|
| `fProxyClement(n)` | symmetric tridiag, **0 diagonal**, off[i]=`√((i+1)(n−1−i))` | **eigenvalues exactly {n−1, n−3, …, −(n−1)}**; trace 0 |
| `fProxyWilkinsonPlus(n)` (n odd) | sym tridiag, diag[i]=`|(n−1)/2 − i|`, off=1 | top two eigenvalues nearly equal (near-pair) |
| `fProxyFiedler(n)` | `|i−j|` | symmetric; **one positive eigenvalue, n−1 negative**; det=(−1)ⁿ⁻¹(n−1)2ⁿ⁻² |
| `fProxyDingDong(n)` | symmetric Hankel `0.5/(n − i − j − 0.5)` | eigenvalues in (−π/2, π/2), cluster near ±π/2 |
| `fProxyFrank(n)` | upper Hessenberg: `n−max(i,j)` for `i≤j+1`, else 0 | **det=1**; eigenvalues real, positive, reciprocal pairs; ill-cond |
| `fProxyVandermonde(in fProxyN nodes)` | `nodes[i]^j`, square n×n | **det=∏_{i<j}(nodes[j]−nodes[i])** |
| `fProxyCompanion(in fProxyN coeffs)` | companion of monic `xⁿ+Σc_k xᵏ` (sub-diag 1, last col `−c`) | **eigenvalues = polynomial roots** |
| `fProxyHadamard(n)` | Sylvester ±1 (n power of 2, else throw) | **HᵀH = nI** (orthogonal·√n); cond=1; \|det\|=n^{n/2} |
| `fProxyCirculant(in fProxyN c)` | `C[i,j]=c[(j−i) mod n]` | **eigenvalues = DFT(c)** → cross-check the library FFT |
| `fProxyKahan(n, θ)` | upper-tri `S·R`, S=diag(sᵏ), R=I−c·(strict upper), s=sinθ,c=cosθ | ill-conditioned; classic QRCP "no-pivot" counterexample |
| `fProxyTriw(n, α)` | upper-tri, 1 on diag, α on every super-entry | **det=1**, all **eigenvalues=1**, ill-cond for α≪0 |
| `fProxyLauchli(n, ε)` | (n+1)×n: row0=ones, rows1..n = ε·I | full col rank but near-deficient — QR-vs-SVD LS stress |

---

## Tests (test-writer, after review)
Two kinds, both valuable:
1. **Property tests** — each generator's documented closed form: Pascal/MinIJ/Moler/Triw `det≈1`;
   Laplacian eigenvalues vs `2−2cos(kπ/(n+1))` and det=n+1; KMS det=(1−ρ²)ⁿ⁻¹; Pei eigenvalues {α+n, α…};
   Clement eigenvalues {n−1,…,−(n−1)}; Fiedler one-positive-rest-negative + det; Hadamard `HᵀH==nI`;
   Vandermonde det = ∏ node differences; Companion eigenvalues = chosen roots; Lehmer/Hilbert/Moler
   Cholesky-succeeds + symmetric; **Circulant eigenvalues == library `fft` of the first column** (the
   cross-module check).
2. **Algorithm-exercise tests** — feed the generators into the existing solvers as honest inputs:
   CG on Laplacian1D; `eigenvaluesQR` on Frank (real positive) and Companion (roots); SVD/`pinvSolve`
   vs plain QR on Läuchli (the rank-stress comparison); QRCP on Kahan. These replace the inline
   constructions currently in `LiteratureTests`.
- float + double generated variants; per-precision tolerance via `Consts.fProxySqrtEps` (ill-conditioned
  ones — Hilbert/Frank/Kahan — need loose float / tight double, or double-only tight asserts).

## Deferred (Phase 2 — easy adds once the pattern is set)
Rosser (fixed 8×8, double + near + zero eigenvalues), Magic (row/col sums, rank rules), Redheffer
(det = Mertens), GCD/Fibonacci (det = ∏φ), Lotkin (nonsym Hilbert), Parter/Prolate (Toeplitz, σ near π),
Grcar (pseudospectra), Cauchy (general), Chebyshev-Vandermonde. Int-typed variants of the integer
matrices. quaternion/2D-Poisson block matrices.

## Build order
1. Batch A + Batch B coders in parallel (disjoint files, one partial class), no regen.
2. One regen pass → 3-agent review → fix.
3. test-writer (property + algorithm-exercise) → full suite green.
4. README "Test matrices / gallery" feature line + memory note.
