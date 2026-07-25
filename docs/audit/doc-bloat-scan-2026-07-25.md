# Doc-bloat scan — 2026-07-25

> **RESOLVED 2026-07-25.** 21 of the 25 findings applied; suite 7128/7128. Two deliberate
> overrules, both KEEP:
>
> - **`OP/QP.fProxy.cs:18-19`** — the `½` in `½xᵀQx + cᵀx` is a *contract*, not exposition: a
>   caller who assumes `xᵀQx` scales Q by 2 with no error, and unlike `LP.solve` the objective
>   shape is not inferable from the parameter list. The `k <= n, rows independent` line is a real
>   precondition. Both stay.
> - **All four demo findings** (`Truss3DStabilityDemo.cs` ×3, `BuildingFrameStabilityDemo.cs` ×1,
>   13 lines) — scoping error in the brief, which put `Assets/Demos/**` in scope. Demos are
>   teaching material by design; "why IC(0) beats block-Jacobi here" is what a demo is *for*.
>   Do not re-flag; fix the brief before re-running this scan.
>
> The `QP.fProxy.cs:21-25` derivation was cut but its Nocedal & Wright citation kept as a
> one-liner — provenance is cheap and the "no iteration loop" fact is load-bearing.

Read-only scan for domain-exposition bloat: comments and XML docs that teach the reader the
subject matter instead of stating the contract of the code. Per
`docs/dev/brief-doc-bloat-scan.md`. This is NOT the jargon sweep and NOT the dev-history sweep
(both already done separately) — it hunts prose that is correct and well-written but unnecessary.

**Scope scanned**: `Assets/LinearAlgebra/CodeGen/TemplateSource/**` (~280 template files, every
file), `Assets/LinearAlgebra/Benchmarks/*.cs` (40 files, 4,080 lines), `Assets/Demos/**` (20
files, 5,040 lines), `docs/features/*.md` (22 files, 1,209 lines). `Source/`,
`SourceTests/Generated/`, `Benchmarks/Generated/` excluded (codegen output — never cited).
`DEVLOG.md`, `docs/dev/`, `docs/audit/` excluded (exempt by design).

**Headline result**: this codebase has already been through a serious documentation-discipline
pass (project memory records a prior jargon sweep, a dev-history sweep, and a ~945-line
doc-reduction pass). Only **83 lines** of genuine subject-matter exposition were found across the
entire scanned corpus — well under 0.2% of scanned lines. The Krylov family (40 files, 11,386
lines), Eigen/SVD/LOBPCG/ML, and the ~171 remaining OP/other-folder template files are
essentially clean. What remains is concentrated in `LP.fProxy.cs` (the brief's own calibration
source — its current text literally *is* the calibration example), `QP.fProxy.cs`, one demo file
with a recurring "why this preconditioner wins" habit, and small math-restates-signature intros
in `docs/features/*.md` and two benchmark files.

## Findings

Ordered by lines-saved descending.

| file:line | category | lines | verdict |
|---|---|---|---|
| `docs/features/lp-lad.md:3-9` | math-restates-signature | 7 | CUT |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.fProxy.cs:13-18` | math-restates-signature | 6 | CUT |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.fProxy.cs:21-26` | derivation | 6 | CUT-partial |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs:21-25` | derivation | 5 | CUT-partial |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs:630-634` | math-restates-signature | 5 | CUT |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.fProxy.cs:46-49` | benchmark-verdict-in-public-doc | 4 | CUT-partial |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.fProxy.cs:301-304` | benchmark-verdict-in-public-doc | 4 | CUT-partial |
| `docs/features/qp-mip.md:8-11` | math-restates-signature | 4 | CUT |
| `docs/features/qp-mip.md:27-30` | math-restates-signature | 4 | CUT |
| `Assets/Demos/12_Truss3D/Truss3DStabilityDemo.cs:19-22` | derivation | 4 | CUT-partial |
| `Assets/LinearAlgebra/Benchmarks/TallWideSolveBenchmark.cs:20-23` | math-restates-signature | 4 | CUT-partial |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.fProxy.cs:24-26` | modelling-tutorial | 3 | CUT-partial |
| `docs/features/solvers.md:18-20` | dev-history-in-public-doc | 3 | CUT-partial |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Control.fProxy.cs:9-11` | math-restates-signature | 3 | CUT-partial |
| `Assets/Demos/12_Truss3D/Truss3DStabilityDemo.cs:424-426` | problem-definition | 3 | CUT-partial |
| `Assets/Demos/12_Truss3D/Truss3DStabilityDemo.cs:445-447` | problem-definition | 3 | CUT-partial |
| `Assets/Demos/13_BuildingFrame/BuildingFrameStabilityDemo.cs:20-22` | motivation | 3 | CUT-partial |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs:18-19` | math-restates-signature | 2 | CUT |
| `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.fProxy.cs:294-295` | motivation | 2 | CUT |
| `Assets/LinearAlgebra/Benchmarks/LPBenchmark.cs:101-102` | math-restates-signature | 2 | CUT-partial |
| `Assets/LinearAlgebra/Benchmarks/QPBenchmark.cs:54-55` | math-restates-signature | 2 | CUT-partial |
| `docs/features/decompositions.md:34` | dev-history-in-public-doc | 1 | CUT-partial |
| `Assets/LinearAlgebra/Benchmarks/LPBenchmark.cs:131` | math-restates-signature | 1 | CUT-partial |
| `Assets/LinearAlgebra/Benchmarks/CholeskyBenchmark.cs:5` | math-restates-signature | 1 | CUT-partial |
| `Assets/LinearAlgebra/Benchmarks/FittingBenchmark.cs:19` | problem-definition | 1 | CUT-partial |

### `docs/features/lp-lad.md:3-9` — math-restates-signature (7 lines, CUT)
```
Solves linear programs in canonical primal form

minimize    cᵀx
subject to  Aᵢ·x {≤, =, ≥} bᵢ   (per-row sense)
            x ≥ 0
```
Exact CUT-calibration pattern. Fully redundant with prose later in the same file (line 31-33:
"Variables are non-negative by construction..."). No residue needed.

### `OP/LP.fProxy.cs:13-18` — math-restates-signature (6 lines, CUT)
```
    // Canonical primal form solved by the public entry points:
    //
    //     minimize    cᵀx
    //     subject to  Aᵢ·x  {≤, =, ≥}  bᵢ    (per-row sense in `senses`)
    //                 x ≥ 0
    //
```
This is literally the brief's own calibration example — restates `LP.solve`'s own signature/param
docs as math. The immediately-following backend bullet list (lines 19-22, real routing guidance)
should stay. No residue needed.

### `OP/Kalman.fProxy.cs:21-26` — derivation (6 lines, CUT-partial)
```
// steadyStateGain reuses Control's own SDA (structure-preserving doubling) DARE engine under the
// LQR/KF DARE DUALITY: the filter's predicted-covariance DARE
//     Σ = AΣAᵀ + Q - AΣHᵀ(HΣHᵀ+R)⁻¹HΣAᵀ
// is exactly Control's LQR DARE S = Q + ÃᵀSÃ - ÃᵀSB̃(R+B̃ᵀSB̃)⁻¹B̃ᵀSÃ under Ã=Aᵀ, B̃=Hᵀ (S↔Σ) --
// i.e. Riccati.dare(Aᵀ, Hᵀ, Q, R, ...) IS this filter's steady-state Riccati solve. No second
// Riccati implementation exists in this file.
```
Proves the substitution makes the two Riccati equations identical — a correctness proof for a
maintainer, not something a caller needs. Residue: "steadyStateGain reuses Control's LQR/KF DARE
duality (`Riccati.dare(Aᵀ, Hᵀ, Q, R, ...)`) — no second Riccati implementation exists in this
file."

### `OP/QP.fProxy.cs:21-25` — derivation (5 lines, CUT-partial)
```
    // Q is symmetric PSD (v1 contract): a singular reduced Hessian is regularized (δ·‖Q‖∞·I retry)
    // rather than handled via negative-curvature machinery. Indefinite Q is out of scope (NP-hard in
    // general). One null-space Newton step is exact for this problem (Nocedal & Wright eq.
    // 16.16-16.19): substituting x = x0 + Zy for an orthonormal null(A_W) basis Z reduces it to an
    // unconstrained quadratic in y, which Newton's method solves in one step from any start.
```
The last two-and-a-half sentences derive *why* one Newton step suffices. Residue: "Q is symmetric
PSD (v1 contract); a singular reduced Hessian is regularized via a δ·‖Q‖∞·I retry. Indefinite Q is
out of scope."

### `OP/QP.fProxy.cs:630-634` — math-restates-signature (5 lines, CUT)
```
        // Problem solved:
        //
        //     minimize    1/2 xᵀQx + cᵀx
        //     subject to  A x {<=,=,>=} b     (per-row senses, LP.solve's ConstraintSense)
        //                 xl <= x <= xu
```
Duplicates `qpActiveSetCore`'s own XML `<summary>` a few lines below. No residue needed.

### `OP/LP.fProxy.cs:46-49` — benchmark-verdict-in-public-doc (4 lines, CUT-partial)
```
        /// <param name="method">Backend (default RevisedSimplex — fastest exact backend at every
        /// benchmarked size on cold solves and the fastest infeasibility certifier (1-2 pivots);
        /// pick <see cref="LPMethod.DualSimplex"/> explicitly for re-solves from a near-dual-feasible
        /// state, <see cref="LPMethod.InteriorPoint"/> for very ill-conditioned vertices.</param>
```
A benchmark verdict baked into a public XML doc — forbidden by CLAUDE.md and rots the moment
anything is re-benchmarked. Residue: "default RevisedSimplex; pick DualSimplex for warm re-solves,
InteriorPoint for very ill-conditioned vertices."

### `OP/LP.fProxy.cs:301-304` — benchmark-verdict-in-public-doc (4 lines, CUT-partial)
```
        /// interior point) above it. BR's weighted-median long step wins at small-to-moderate m (near-
        /// constant, few-microsecond latency -- the common low-observation-count case); FN's fixed
        /// ~10-iteration n×n normal solve wins once m grows large enough that BR's per-pivot sweep over
        /// m rows dominates. The crossover is a measured, re-tunable, PER-DTYPE value (see the comment
```
Same pattern inside `lad`'s hybrid-routing doc — a "which is faster and why" narrative rather than
a routing fact. Residue: "Routes to `ladBR` at/below the per-dtype threshold (see dispatch below),
`ladFN` above it; call either directly to bypass routing."

### `docs/features/qp-mip.md:8-11` and `:27-30` — math-restates-signature (4 + 4 lines, CUT)
```
minimize    ½xᵀQx + cᵀx
subject to  Aᵢ·x {≤, =, ≥} bᵢ,   xl ≤ x ≤ xu
```
```
minimize    cᵀx
subject to  Aᵢ·x {≤, =, ≥} bᵢ,   xl ≤ x ≤ xu,   xⱼ ∈ ℤ for flagged j
```
Both immediately precede the `QP.solve`/`MIP.solve` signature lines, which already name and
constrain every symbol used. No residue needed.

### `Assets/Demos/12_Truss3D/Truss3DStabilityDemo.cs:19-22` — derivation (4 lines, CUT-partial)
```
/// The preconditioner is switchable at runtime (block-Jacobi / IC(0) / SSOR) and the cold
/// iteration count is displayed so their strength is directly comparable: on the slender
/// tower IC(0) and SSOR capture the inter-story coupling that resolves the global
/// sway/torsion mode — the softest eigenvector — while block-Jacobi sees only each node's own
/// diagonal block and needs several times as many iterations to reach it.
```
The "why IC(0)/SSOR converge faster" clause is general preconditioner theory. Residue: "The
preconditioner is switchable at runtime (block-Jacobi/IC(0)/SSOR) and the cold iteration count is
displayed so their strength is directly comparable."

### `Assets/LinearAlgebra/Benchmarks/TallWideSolveBenchmark.cs:20-23` — math-restates-signature (4 lines, CUT-partial)
```
//   TALL  (m = 2n): the OVERDETERMINED least-squares problem min ||A x - b|| via Householder QR
//         (decompInPlace forms the thin Q; solveInPlace does the direct no-Q solve).
//   WIDE  (n = 2m): the UNDERDETERMINED minimum-norm problem min ||x|| s.t. A x = b via LQ
//         (decomp A = L Q; minNormSolve x = Qᵀ L⁻¹ b), plus the row-pivoted rank-revealing LQRP.
```
The `min ||Ax-b||` / `min ||x|| s.t. Ax=b` formulas and the `x = Qᵀ L⁻¹ b` derivation restate
textbook LS/min-norm forms. Residue: "TALL (m=2n) exercises decompInPlace/solveInPlace via QR;
WIDE (n=2m) exercises LQ decomp/minNormSolve, plus row-pivoted rank-revealing LQRP."

### `OP/LP.fProxy.cs:24-26` — modelling-tutorial (3 lines, CUT-partial)
```
    // L1 regression (least absolute deviation) is the flagship application: minimize ‖Ax − b‖₁ over a
    // FREE x is exactly an LP once each residual is split into a +/− pair (see `lad`). A fast,
    // approximate iteratively-reweighted-least-squares alternative lives in Optimize.ladIRLS.
```
Residue: "L1 regression is available via `lad`; see `Optimize.ladIRLS` for an approximate
alternative."

### `docs/features/solvers.md:18-20` — dev-history-in-public-doc (3 lines, CUT-partial)
```
`CHO.solveInPlace(ref A_to_L, ref b_to_x)` fuses `decompInPlace`+`decompSolve`; `A_to_L` exits as
a usable factor. (The old `choleskySolve(in A, ref L, ref b)` two-line composition-in-disguise
was deleted — write the explicit `CHO.decomp` + `CHO.decompSolve` composition if `A` must survive.)
```
A removed-API postmortem in a user-facing doc. Residue: "write the explicit `CHO.decomp` +
`CHO.decompSolve` composition if `A` must survive."

### `OP/Control.fProxy.cs:9-11` — math-restates-signature (3 lines, CUT-partial)
```
// Discrete-time LQR. x_{k+1} = A x_k + B u_k, cost Σ(xᵀQx + uᵀRu), optimal feedback u = -Kx via the
// discrete algebraic Riccati equation (DARE)
//     S = Q + AᵀSA - AᵀSB(R+BᵀSB)⁻¹BᵀSA,   K = (R+BᵀSB)⁻¹BᵀSA.
```
Canonical LQR statement restated as math; the per-method XML docs already state the `u=-Kx` sign
convention and Q/R meanings. Residue: "Discrete-time LQR: optimal feedback u=-Kx from the DARE,
solved via `Riccati.dare` (SDA)."

### `Assets/Demos/12_Truss3D/Truss3DStabilityDemo.cs:424-426` — problem-definition (3 lines, CUT-partial)
```
/// preconditioner. IC(0) factors A's lower block pattern, so it carries the inter-story
/// coupling that resolves the tower's global sway/torsion mode — the softest eigenvector,
/// which the diagonal-only block-Jacobi of the 2D house frame cannot see.
```
General preconditioner-theory justification on a one-line `Execute`. Residue: "Warm LOBPCG
smallest-k eigenpairs of the tower stiffness matrix with an IC(0) preconditioner."

### `Assets/Demos/12_Truss3D/Truss3DStabilityDemo.cs:445-447` — problem-definition (3 lines, CUT-partial)
```
/// preconditioner (omega=1, symmetric Gauss-Seidel). Like IC(0) it carries inter-story
/// coupling through its forward/backward sweeps, but as a stationary iteration rather than a
/// factorization it needs roughly twice IC(0)'s iterations on this pencil.
```
Same pattern for SSOR. Residue: "Warm LOBPCG smallest-k eigenpairs of the tower stiffness matrix
with an SSOR preconditioner (omega=1, symmetric Gauss-Seidel)."

### `Assets/Demos/13_BuildingFrame/BuildingFrameStabilityDemo.cs:20-22` — motivation (3 lines, CUT-partial)
```
/// building: at 8×8 bays × 40 stories the system is ~10k dof, the regime where the
/// preconditioner choice actually bites — IC(0)'s forward/backward triangular solve is
/// serial, block-Jacobi's diagonal apply is fully parallel, and the cold-iteration readout
```
"IC(0) serial vs block-Jacobi parallel" is a general computational-complexity fact about the
algorithm classes, not this demo. Residue: "This is the tower demo scaled to a real building: ~10k
dof at 8×8 bays × 40 stories, and the cold-iteration readout lets you watch preconditioner choice
matter as you switch."

### `OP/QP.fProxy.cs:18-19` — math-restates-signature (2 lines, CUT)
```
    //     minimize    ½xᵀQx + cᵀx   subject to   A_W x = b_W
    //     (A_W: k x n, k <= n, rows independent -- "the working set")
```
Redundant with `eqpSolve`/`eqpNullSpaceStep`'s own `<param>` docs below, which already state
"Working-set constraint matrix, k x n (1 <= k <= n), rows independent."

### `OP/LP.fProxy.cs:294-295` — motivation (2 lines, CUT)
```
        /// Least absolute deviation (L1 regression): minimize ‖A x − b‖₁ over a FREE x ∈ ℝⁿ. Robust to
        /// outliers where ordinary least squares (which minimizes the L2 norm) is not. This overload
```
"Robust to outliers where ordinary least squares... is not" is textbook motivation for choosing L1
over L2, not needed to call `lad` correctly.

### `Assets/LinearAlgebra/Benchmarks/LPBenchmark.cs:101-102` — math-restates-signature (2 lines, CUT-partial)
```
//   Section 1 (LP.solve): random dense feasible+bounded LPs (min cᵀx s.t. A x <= b, x >= 0, A >= 0,
//     b = A x0 + slack so x0 is feasible). THREE backends compared head to head -- Mehrotra interior
```
Restates `LP.solve`'s own standard-form contract. Residue: "random dense feasible+bounded LPs (A
>= 0, b = A x0 + slack so x0 is feasible), three backends compared head to head on the identical
problem."

### `Assets/LinearAlgebra/Benchmarks/QPBenchmark.cs:54-55` — math-restates-signature (2 lines, CUT-partial)
```
//   Section 1 (QP.solve, random SPD QP): random dense feasible+bounded convex QPs (min 1/2 xQx + cx
//     s.t. A x <= b, xl <= x <= xu; Q symmetric PSD via Rand.spdInPlace with a modest condition
```
Restates `QP.solve`'s own standard-form contract. Residue: "random dense feasible+bounded convex
QPs; Q symmetric PSD via Rand.spdInPlace, condition ~10, x boxed in [0,3]."

### `docs/features/decompositions.md:34` — dev-history-in-public-doc (1 line, CUT-partial)
```
`QRCP.solveInPlace(...)` factors A's own buffer directly (no Q scratch, no memcpy — strictly
faster than the old copying form) and leaves A_to_Q as a *usable* orthogonal factor...
```
Residue: "factors A's own buffer directly (no Q scratch, no memcpy)" — drop "than the old copying
form."

### `Assets/LinearAlgebra/Benchmarks/LPBenchmark.cs:131` — math-restates-signature (1 line, CUT-partial)
```
//   Section 4 (dense covering LP): min cᵀx s.t. A x >= b, x >= 0 with A,b,c >= 0 by construction --
```
Opening clause restates canonical covering-LP form. Residue: "dense covering LP with A,b,c >= 0 by
construction, deliberately dual-favorable."

### `Assets/LinearAlgebra/Benchmarks/CholeskyBenchmark.cs:5` — math-restates-signature (1 line, CUT-partial)
```
// Cholesky factorization A = L L^T of a symmetric positive-definite matrix. A is taken `in`
```
Opening clause names the textbook factorization form. Residue: "A is taken `in` (never mutated)
and L is overwritten each run, so the SPD input is built once and every timed sample does
identical work."

### `Assets/LinearAlgebra/Benchmarks/FittingBenchmark.cs:19` — problem-definition (1 line, CUT-partial)
```
// NOT total least squares / total least deviation (those minimize orthogonal distance).
```
The parenthetical defines what TLS is in general. Residue: "Response-residual fitting, NOT total
least squares / total least deviation."

## Clean areas (scanned in full, nothing to report)

- **All of `OP/Krylov*.cs`** (40 files, 11,386 lines): scalar solvers, rectangular/least-norm
  family, shared cores, and the entire `Krylov.Block.*` family. Every precondition, buffer
  contract, and routing note is load-bearing — this folder has many near-duplicate solvers that
  differ only in which quantity they minimize/characterize, so the one-line math
  characterizations are non-obvious-overload-routing information, not exposition.
- **`OP/Eigen*.cs`, `OP/SVD*.cs`, `OP/LOBPCG*.cs`, `ML/*.cs`**, and most of `OP/Kalman*.cs`/
  `MPC*.cs`/`Riccati*.cs`/`NLS*.cs`: contract-only, including places that would tempt exposition
  (UKF sigma-point formulas, randomized-SVD sketch math, LOBPCG convergence tests).
- **The remaining ~171 `OP/*.cs` and other-top-level-folder template files** — QR/QRCP/LQ/LQRP/
  CHO/CHOP/LU/Bidiag/Blas/DetMath/FFT/GenOP/Optimize/Norms/Query*/Random*/Resample/Select/Swap/
  Unsafe*/Wave/WideOP, plus Analysis/, bool/, Debug/, fProxy/, Gallery/, Generate/, Hash/,
  Indices/, Interfaces/, iProxy/, MG/, Pivot/, Realtime/, Sparse/ (full BSR + preconditioner set),
  Statistics/. `Gallery/` was flagged as a historical risk area but is already tight: each
  generator is a precise recipe + the "known property" needed for its role as a test fixture, not
  a definition of what an eigenvalue is. Zero findings.
- **35 of 40 Benchmarks files and 15 of 20 Demos files**, including the largest
  (RooflineBenchmark.cs, HoverTankDemo.cs, DemoSmokeTests.cs, EigenSvdBenchmark.cs,
  SparseSolverBenchmark.cs): methodology/mechanics only.
- **17 of 22 `docs/features/*.md` files**: e.g. `control.md`'s LQR cost-functional statement was
  considered and kept — it is the only place that explains what `Q`/`R` semantically weight, not
  duplicated elsewhere; `svd.md`'s Jordan-Wielandt accuracy note and `eigen.md`'s buckling
  matrix-slot recipe are genuine usage guidance, not tutorials.

## Total and priority

**83 lines flagged for cut**, across 25 findings, out of roughly 62,000 scanned lines
(~0.13%). The three files worth doing first, by lines-saved:

1. **`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.fProxy.cs`** — 19 lines across 5 findings
   (also the brief's own calibration source; the file's current text already matches the
   calibration examples almost verbatim).
2. **`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QP.fProxy.cs`** — 12 lines across 3 findings.
3. **`Assets/Demos/12_Truss3D/Truss3DStabilityDemo.cs`** — 10 lines across 3 findings, all the
   same "why this preconditioner wins" habit (a fourth instance of the identical habit appears in
   `BuildingFrameStabilityDemo.cs`).
