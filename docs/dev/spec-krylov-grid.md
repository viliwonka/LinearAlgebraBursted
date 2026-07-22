# Krylov grid: gallery × preconditioner × solver (task #59)

A single **compatibility grid** over (BSR gallery matrix) × (preconditioner) × (single-RHS square
Krylov solver), exposed TWO ways sharing one runner:
1. **Unit test** — classifies every compatible cell and asserts the honesty/robustness net.
2. **Benchmark** — times each solver per-iteration on the BSR galleries (feeds task #58 optimize).

Scope is deliberately SMALL (user: "don't make it take too much time"). Single-RHS square family
only (that's where the preconditioner cross is richest); block families are out of scope for v1.

## Reuse (do NOT reinvent)
- Invokers: `IfProxySquareSolverInvoker` + the concrete `fProxy{Cg,Fcg,Minres,MinresQLP,BiCGStab,
  Gmres,Fgmres,Idr,Tfqmr,Gcrodr}Invoker` structs in `KrylovBattery.Invokers.fProxy.cs`. Each carries
  `Requires`/`Forbids`/`PrecondKind`/`Tol`/`MaxIter(n)` and a generic
  `SolveWithPrecond<TOp,TPre>(in A, in M, in b, ref x)`. Construct with `{ TolValue=..., MaxIterMul=... }`
  (gmres/fgmres/gcrodr also need `Restart`; idr needs `S`/`Seed`; gcrodr also `Recycle`).
- Compatibility predicate: `MatrixProfileMatch.Applicable(inv.Requires, inv.Forbids,
  GalleryProfiles.Of(gm))` (in `KrylovBatteryProfile.cs`).
- Galleries: `fProxyKrylovBatteryGallery.Build(ref arena, GalleryBSRMatrix)` (BSR).
- Oracle: `fProxyKrylovBatteryOracles.RelResidualBSR(in A, in x, in b)` (fresh true rel-residual).
- Preconditioner factories (see `PreconditionerBatteryTests.fProxy.cs` for exact call shapes):
  `arena.fProxyBlockJacobi/ fProxySSOR / fProxyIC0 / fProxyChebyshev / fProxyFSAI /
  fProxyAdditiveSchwarz / fProxyILU0 / fProxySPAI / fProxyRestrictedSchwarz(in A)`; AMG is
  `var amg = arena.fProxyAMG(in A, out var info); var M = new fProxyAMGPreconditioner(in amg); ...; amg.Dispose();`.
- BSR→operator wrapping (`fProxyBSROperator`): mirror how `KrylovSquareBatteryTests.fProxy.cs` wraps
  a BSR gallery entry before calling an invoker.

## Galleries (3, all BSR so preconditioners apply)
- `GalleryBSRMatrix.Laplacian2D_16x16`   (SPD, n=256)  — primary SPD
- `GalleryBSRMatrix.RandomSparseSPD_120_2` (SPD, n=240) — 2nd SPD, BR=2
- `GalleryBSRMatrix.RandomSparseNonsym_80` (nonsym, n=80) — nonsym

## Preconditioners (11 columns incl. Identity)
Define a grid-local `enum GridPrecond { Identity, BlockJacobi, SSOR, IC0, Chebyshev, FSAI,
AdditiveSchwarz, AMG, ILU0, SPAI, RestrictedSchwarz }`.
- **symmetric-M** (SPD galleries only): BlockJacobi, SSOR, IC0, Chebyshev, FSAI, AdditiveSchwarz, AMG.
- **nonsym-M** (any square gallery): ILU0, SPAI, RestrictedSchwarz.
- **Identity**: any gallery (the unpreconditioned baseline — call the invoker's plain `Solve`, NOT
  SolveWithPrecond, to exercise the real unpreconditioned entry point).

## Cell compatibility (all three must hold, else Skipped)
1. solver vs gallery: `MatrixProfileMatch.Applicable(Requires, Forbids, GalleryProfiles.Of(gm))`.
2. solver vs precond symmetry: a **symmetric solver** (`PrecondKind == SymmetricBSR`: cg/fcg/minres/
   minresQLP) accepts ONLY symmetric-M or Identity. A **nonsym solver** (`PrecondKind ==
   NonsymmetricBSR`: biCGStab/gmres/fgmres/idr/tfqmr/gcrodr) accepts ANY precond.
3. precond vs gallery: symmetric-M preconditioners require an SPD gallery; nonsym-M and Identity
   accept any square gallery.

## Cell outcome (classification)
```
enum CellOutcome { Skipped, Converged, MaxIterations, Errored, FalseConverged }
```
- `Errored`   = status is Breakdown/Diverged/Degenerate, OR x has any NaN/Inf.
- `FalseConverged` = status == Converged but fresh `RelResidualBSR(A,x,b) > bound`  (bound = 100*Tol,
  matching the preconditioner battery's generous band). **This must NEVER happen** — it's the #53
  honesty invariant.
- `Converged`  = status == Converged AND fresh residual ≤ bound.
- `MaxIterations` = status == MaxIterations (ALLOWED — a solver may run out of budget on a hard cell).

Use a generous budget: `MaxIterMul` giving ~8*n iterations (as the precond battery does), `tol = Consts.fProxySqrtEps`.

## Generic dispatch (the one runner)
Put the precond switch INSIDE a method generic over the invoker so TPre is inferred:
```
static CellOutcome RunCell<TInv>(TInv inv, in fProxyBSR A, in fProxyBSROperator op,
                                 in fProxyN b, ref fProxyN x, GridPrecond pk, ref Arena arena)
    where TInv : struct, IfProxySquareSolverInvoker
{
    // zero x; then:
    switch (pk) {
      case Identity: info = inv.Solve(in op, in b, ref x); break;
      case IC0:      { var M = arena.fProxyIC0(in A);       info = inv.SolveWithPrecond(in op, in M, in b, ref x); } break;
      case AMG:      { var amg = arena.fProxyAMG(in A, out _); var M = new fProxyAMGPreconditioner(in amg);
                       info = inv.SolveWithPrecond(in op, in M, in b, ref x); amg.Dispose(); } break;
      ... one arm per GridPrecond ...
    }
    return Classify(info, A, x, b, inv.Tol);
}
```
Outer switch over a `enum GridSolver` picks the concrete invoker type and calls
`RunCell<fProxyCgInvoker>(new fProxyCgInvoker{...}, ...)` etc. TOp is `fProxyBSROperator`.
(One switch of ~10 solver arms + one switch of ~11 precond arms = ~21 arms total, NOT 110.)

## Test file: `KrylovGridTests.fProxy.cs` (TemplateSourceTests/fProxy/)
- Mirror `PreconditionerBatteryTests.fProxy.cs` structure: a `[BurstCompile(CompileSynchronously=true,
  FloatPrecision=High, FloatMode=Default)] struct TestJob : IJob` that loops **all (gallery, solver,
  precond) triples**, classifies each via RunCell, and records the FIRST offending cell into a
  `NativeArray<fProxy> Fail` (flag, encoded gallery/solver/precond indices, got-value). Each cell uses
  its own `new Arena(Allocator.Persistent)` … `arena.Dispose()` (per-cell, like the precond battery).
- Drive with `[TestCaseSource]` over the 3 galleries (one NUnit case per gallery, so a failure names
  the gallery) OR a single `[Test]` looping all — either is fine; per-gallery is nicer for triage.
- **Assertions (generous, regression-focused — NO iteration-count races, per project memory):**
  - PRIMARY: across the whole grid, **no cell is `Errored`** and **no cell is `FalseConverged`**.
    (This is the honest robustness/anti-silent-divergence net; ties to #53.)
  - SECONDARY: on the two SPD galleries, the known-good preconditioned SPD combos converge — assert
    that cg+{IC0,FSAI,AMG,BlockJacobi} all reach `Converged` (these pass in the precond battery, so a
    regression here is real). Do NOT assert Identity or hard nonsym cells converge (they may legitimately
    hit MaxIterations — allowed).
- Record enough in `Fail[]` for `Assert.Fail($"gallery G solver S precond P → outcome O (resid R)")`.

## Benchmark: `KrylovGridBenchmark.fProxy.cs` (codegen half) + `KrylovGridBenchmark.cs` (hand half)
Scope the benchmark to **solver per-iteration cost** (what #58 optimizes), NOT the full precond grid:
- For each of the 2 SPD-or-nonsym BSR galleries compatible with the solver, time the invoker's plain
  `Solve` (Identity) with a FIXED iteration budget and `tol = 0` (forces the full budget for
  deterministic timing — mirror `IterativeBenchmark`'s maxIter=100/tol=0 rationale). Pick a fixed budget
  (e.g. 100) via a maxIter passed to the primitive, not the invoker's 8*n (we want fixed-count timing).
  - NOTE: the invokers derive maxIter internally from `MaxIter(A.Rows)`. For fixed-count timing either
    set `MaxIterMul` so `MaxIterMul*n == desired` is impractical → instead call the `Krylov.*` primitives
    directly in the timed job (as `IterativeBenchmark`'s `CGJobFProxy` does), OR add a fixed-iter timed
    job per solver. Simplest: hand-roll one timed IJob per solver (like CGJobFProxy) calling the
    `Krylov.<solver>` BSR-operator overload with a fixed maxIter, tol=0. Keep to the ~6-9 solvers that
    have a clean BSR+fixed-iter entry.
- Print a table: rows = solver, one column-group per gallery, `Bench.Time`/`Bench.RowTime` style. Mirror
  an existing benchmark's hand-written `Run()`/`Section(sb)` half (e.g. `SparseSolverBenchmark.cs`).
- Register `KrylovGridBenchmark.Section(sb)` in `AllBenchmarks.cs` (after `PCGBenchmark.Section`).
- If the fixed-iter-per-solver timed jobs balloon the benchmark, SCOPE DOWN to cg/minres/biCGStab/gmres/
  idr only and note the rest as follow-up — the goal is a representative per-iteration cost baseline for
  #58, not exhaustive coverage.

## Process / acceptance
- Templates are source of truth; regen via `Tools/regen.ps1`; never hand-edit Generated. New top-level
  types must carry the `fProxy` token (CS0101 across float/double in one assembly) — mirror the existing
  invoker naming note.
- Run the FULL suite in the FOREGROUND to green before committing. Baseline is **7123 passed** — the new
  grid test adds cases, so the count RISES; report the exact new total. Zero failures.
- Benchmark must compile and appear in the AllBenchmarks report (a quick benchmark run to confirm it
  emits a section is enough; no perf tuning here — that's #58).
- Commit locally on green (do NOT push — that's the session cadence for coder output; I push after review).
  Message: "Krylov grid: gallery×preconditioner×solver compatibility test + solver benchmark (#59)".
- 1-hour park rule: if stuck >1h, commit WIP or park with a note and report.
```
