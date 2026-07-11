# Spec: Barrodale–Roberts exact LAD (second exact engine)

Goal: the classical specialized-simplex LAD solver as `LP.ladBR` — the second reformulation-free
exact engine beside Frisch–Newton (`ladFN`), the small-m champion, and an independent oracle for
FN's tests. Definition of done: `docs/dev/spec-shipped-feature.md`.

## Why a second exact engine

BR and FN fail differently (combinatorial pivoting vs interior-point convergence), so agreement
between them is strong evidence of correctness — a permanent cross-check fixture. Literature
(Portnoy–Koenker 1997): BR wins small-to-moderate m, FN wins large m (crossover ~10³–10⁴). BR also
returns an exact VERTEX solution (residuals exactly zero at n points), which FN's interior path
only approaches.

## Algorithm — transcribe from a fetched source (mandatory)

Barrodale & Roberts 1973, "An improved algorithm for discrete l1 linear approximation";
reference implementations: ACM TOMS Algorithm 478 (Fortran), R quantreg `rqbr.f`
(Koenker–d'Orey adaptation, also does quantiles). Fetch one (cdn.jsdelivr.net mirrors of
cran/quantreg worked before; raw.githubusercontent may rate-limit) and transcribe — do NOT
reconstruct from memory. Record the source + any deviations in the file header.

Shape of the method (for orientation only, the source is authoritative): a primal simplex on the
LAD LP's special structure, worked directly on an (m)×(n+2)-ish condensed tableau of the ORIGINAL
data — no 2n+2m variable splitting. Its signature trick: in the ratio test it passes THROUGH
vertices whose reduced-cost change doesn't flip the sign of the objective gradient — a
weighted-median line search along the entering direction — so one iteration crosses many
breakpoints; iteration counts scale ~O(n), not O(m). Two stages in the 1973 paper (basis of
observations in stage 1, interchanges in stage 2).

## API

- `LP.ladBR(in fProxyMxN A, in fProxyN b, ref fProxyN x, out double objective, int maxIter = 0)` —
  exact LAD; objective = honest recomputed ‖Ax−b‖₁ (same convention as ladFN). LPInfo result
  (Optimal / MaxIterations).
- If the fetched source is the quantile-capable rqbr variant and τ falls out naturally, add the
  `tau` overload mirroring ladFN's; if it complicates the transcription, τ=0.5 only and note it.
- `LP.lad`'s default stays FN. BR is an explicit engine + test oracle.

## Numerics / repo constraints

Fully fProxy-templated (no double-only kernel, no literal float; double for pure-local scalar
accumulators only); per-dtype tolerances via Consts, mirroring LP.RevisedSimplex conventions;
Allocator.Temp everything, disposed on all paths; job-safe; CS1750/CS0111 pitfalls per
LP.RevisedSimplex.fProxy.cs headers. New file:
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.BarrodaleRoberts.fProxy.cs`.

## Tests (template suite, both dtypes, Burst-executed compute)

1. BR vs ladFN vs LPMethod.Simplex on the benchmark's random-outlier construction, m ∈ {48, 96, 192},
   n = 4: residual agreement 1e-6 rel double / 1e-2 float (float bound per FN-test precedent).
2. Stackloss literature vector (existing data/expected values — reuse).
3. A second literature vector BR-specific if the fetched source ships a worked example (several
   externally-sourced answers is the spec-shipped-feature target).
4. Vertex property: at the BR optimum, at least n residuals are exactly ~0 (|r| ≤ tight per-dtype
   tol) — the property FN cannot certify; this is BR's distinguishing test.
5. Degenerate exact-fit (b = A·x_true): residual ~0, terminates.
6. Failure case: maxIter=1 returns MaxIterations with a usable best iterate, no leaks/NaN.

## Benchmark

One `LP.ladBR` row in the LAD section at the existing sizes, same job pattern (outputs from inside
the Burst job, honest recomputed residual). Expectation to verify: BR competitive with or beating
FN at m=48, losing by m=384. States its budget share; LAD section must stay comfortably inside the
repo's ≤10-minute total.
