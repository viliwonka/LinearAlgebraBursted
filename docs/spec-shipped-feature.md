# Spec: what "shipped" means — the definition of done for every feature

A feature is SHIPPED only when all five pillars below hold. This is the per-task standard for all
work, attended or autonomous. Partial states are fine mid-flight, but a feature is not announced,
defaulted-to, or checked off a TODO until it clears this list.

## 1. Naming / syntax — consistent with the library

- Names follow `docs/naming-style-guide.md` and existing precedent. Check precedent BEFORE inventing:
  solver classes use the 4-token grid (`decomp` / `decompInPlace` / `decompSolve` / `solveInPlace`),
  transpose variants use the `TransA` suffix (`matMatDotTransA`, `decompSolveTransA`), workspaces are
  `*Cache`/`*Workspace`, struct-functor interfaces are `IfProxy*`.
- The `fProxy` token appears only on data types/factories per the naming rules; a class = a
  factorization or a coherent capability, never a hollow namespace consuming another class's output.
- New PUBLIC names not covered by precedent require explicit user approval — during autonomous work,
  log to `docs/pending-decisions.md` with a recommendation and skip; never guess.
- Template discipline: `fProxy` is the codegen token; literal `float` is forbidden in templates;
  literal `double` only for pure-local scalar accumulators; `/*+choose[a|b]*/` for per-dtype literals;
  CS1750 (no proxy-typed defaults → forwarding overloads) and CS0111 (no fProxy-returning
  parameterless helpers → inline) respected.

## 2. DRY — reuse before writing

- Reuse library kernels (`LU`, `CHO`, `CHOP`, `QR`, `SVD`, `Blas`, `Krylov`, `UnsafeOP`) instead of
  hand-rolling. A hand-rolled numerical kernel inside a feature file is a smell (the revised simplex's
  private `SolveTranspose` was one; it got promoted into `LU`).
- If a feature needs a primitive the library lacks, the primitive goes into the RIGHT layer
  (`UnsafeOP` for raw SIMD loops, `Blas` for typed kernels, the owning solver class for
  factorization-level operations) as reusable API — with its own tests (pillar 3) — and the feature
  calls it there.
- One template is the single source; cross-dtype sharing uses the established mechanisms (singular
  files, shared TemplateSource types), never copy-paste per dtype.

## 3. Tested

- Unit tests in the template test suite, green in BOTH dtypes, following the house pattern
  (enum + dispatch, `[BurstCompile(CompileSynchronously = true)]`, `Assert.IsTrue` with `==` +
  `Fail[]` diagnostics; template tests probe PUBLIC API only — the firstpass assembly has no
  InternalsVisibleTo; internal-access tests are hand-written in SourceTests/).
- **Burst-compatible, Burst-executed**: the feature itself must compile and run under Burst (job-safe,
  no managed allocations), and the COMPUTE in unit tests and benchmarks must execute inside
  `[BurstCompile(CompileSynchronously = true)]` jobs — never in managed (Mono) code. Mono is 30–40×
  slower and numerically knife-edge-different in float; silent Mono fallbacks once turned the 49s
  suite into 42 minutes, and managed reporting solves once turned a 3-minute benchmark into 13.
  Managed code in tests/benchmarks is for orchestration and assertion plumbing only.
- **Known test vectors from the literature** (search the internet: papers, LAPACK/NIST docs, classic
  datasets — e.g. Stackloss for LAD, Wilkinson/Hilbert matrices; repo precedent: LiteratureTests,
  Gallery). SEVERAL externally-sourced known answers per feature is the target; one is the bare
  minimum; cite each source in a comment.
- **Known failure cases**: what should the feature do on singular / rank-deficient / degenerate /
  infeasible / unbounded / NaN-poisoned / empty input? Each documented behavior gets a test. The
  false-Unbounded phase-1 bug survived 132 unit tests and fell to one adversarially SHAPED input —
  include structure-extreme cases (all-rows-degenerate, exact-fit, duplicated rows), not just random.
- New `Blas`/`UnsafeOP` kernels get DIRECT tests against a plain scalar reference implementation, not
  just indirect coverage through callers (UnsafeOP currently has only indirect coverage — every new
  kernel must not add to that debt).
- Reference-implementation fidelity: algorithms transcribed from a fetched source (paper, reference
  code), never reconstructed from memory; the source is named in the file header.

## 4. Benchmarked

- Timing = Burst `IJob.Run()` (1 warmup + 4 timed, median); ALL reporting data (objective, iterations,
  status, residuals) comes out of the Burst job via output arrays — no managed solves anywhere.
- Honest metrics: recompute reported quality numbers (residuals etc.) from the returned solution
  inside the job; a non-converged run must show a bad number, never an impossible good one.
- Include HARD cases, not just friendly ones: ill-conditioned, structure-adversarial, failure-mode
  rows (status columns where relevant). Benchmarks are how bugs like PC-equilibration over-shrink and
  the phase-1 false-Unbounded were caught — a benchmark that can't fail is decoration.
- **SHORT**: the TOTAL of all benchmarks in the repo must stay under 10 minutes wall-clock. Every new
  section states its budget in a comment; sizes are chosen to inform, not to impress; per-section
  size arrays are independent (no shared arrays across unrelated sections). Any single run
  exceeding ~5 minutes is a bug in the benchmark.

## 5. Optimized

- Level-1: SIMD-friendly inner loops (unit stride, the 2×`fProxy4` accumulator pattern for
  reductions), cache-local access, register reuse. Level-2: unrolling, cache blocking. Level-3:
  GEMM-shaped blocked updates (compact-WY, right-looking panels) where the operation admits them.
- Formulation follows storage: row-major data wants axpy-shaped sweeps for some transposed/column
  operations — pick the variant that keeps memory access contiguous (see
  `docs/perf-vectorization-lessons.md`).
- In-place variants wherever the API grid calls for them; `Allocator.Temp` scratch, disposed on all
  paths; zero managed allocations.
- Algorithmic tricks first, micro-optimization second: the biggest wins in this library's history were
  algorithm-level (revised vs tableau simplex 80×, Frisch-Newton vs LP-reformulation 200×) — check
  the literature for the better algorithm before tuning the worse one.
- Optimization claims require BEFORE/AFTER benchmark numbers on the same harness; results that don't
  reproduce a win get reverted and the negative result recorded (repo precedent: FFT SIMD/radix-8,
  ω-init). Numerical output stays bit-identical, or the tolerance change is documented and justified.

## Process wrapper (how a feature moves through the pillars)

spec (`docs/spec-*.md`, with fetched references) → implement (reuse-first) → tests (pillar 3) →
full suite green → benchmark (pillar 4) → optimize if the numbers say so (pillar 5) → adversarial
review for numerical cores → memory/TODO ledgers updated → atomic commit(s), repo message style.
