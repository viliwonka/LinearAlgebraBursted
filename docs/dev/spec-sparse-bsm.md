# Spec — Block Sparse Matrix (BSM) & Sparse Solvers

Status: **DRAFT / proposal** (2026-07-01). Research-backed design for the first sparse feature in
this library. Non-symmetric first, symmetric mode as a high-value early add. Vectors stay **dense**
(no sparse vectors — deliberate simplification).

> Research provenance: three cited research sweeps (games/sim use-cases + storage formats;
> solvers/preconditioners/orderings/eigensolvers; codebase-convention sweep). Key sources noted inline.

---

## 0. Executive summary (the opinionated version)

- **Format:** BSR / BCSR — "CSR of dense blocks". This is what cuSPARSE, Intel MKL, SciPy, and NVIDIA
  Warp all converge on. A flat **SoA** values buffer (`nnzb · BR · BC` contiguous), plus two index
  arrays (`rowPtr`, `colInd`). One column index per *block*, not per scalar.
- **Blocks are uniform size, but rectangular** `BR × BC` (two ints). Square (`BR == BC`) is the
  simulation/node case; rectangular covers constraint Jacobians and over/under-determined
  least-squares — honoring the "non-square blocks" requirement. **Variable per-node block sizes are
  out of scope for v1** (huge complexity jump; every mainstream lib fixes one block size too).
- **The 3×3 block is the workhorse.** Nearly every real-time deformable system (cloth, mass-spring,
  FEM soft body, projective/backward-Euler dynamics) is a `3n×3n` matrix of `3×3` blocks,
  **symmetric SPD** (sometimes after PSD projection). 2×2 = 2D; 1×1 = scalar PDE/graph Laplacian;
  6×6 = articulated/spatial-inertia. So special-case unrolled kernels for **b ∈ {1,2,3,4,6}**.
- **Symmetric upper-block storage is a clean ~2× win** (memory *and* single-threaded FLOPs). The
  usual multithreaded hazard (the transpose scatter write `y_j += A_ijᵀ x_i` races) **does not exist
  in our single-threaded-per-job Burst core.** High value, include early.
- **Assemble in COO/builder → convert to BSR once → reuse the index structure, overwrite block
  values every frame.** Compressed BSR is expensive to *re-pattern* but trivial to *re-value*. This
  is exactly the FEM/graph repeated-solve loop.
- **Solvers (matrix-free, dense RHS):** CG + block-Jacobi for SPD (the default), MINRES for
  symmetric-indefinite, BiCGSTAB for non-symmetric, CGLS/LSQR for rectangular least-squares, block
  Gauss-Seidel/SOR for the games-physics style. Truncated/warm-started — the frame budget, not the
  √κ convergence bound, sizes the iteration count.
- **Eigen:** power iteration (dominant / PageRank / spectral radius) first; **LOBPCG** for a few
  *smallest* SPD eigenpairs (Fiedler vector → mesh partitioning, vibration modes, λ_min → stability).
  Lanczos (with partial reorthogonalization) optional later. All matvec-only.

---

## 1. Where sparse block systems come from (drives the design)

| System | Block | Dim | Sym? | Definite | Solver |
|---|---|---|---|---|---|
| Cloth / mass-spring (implicit) | **3×3** | 3n | yes | **SPD** | (P)CG |
| FEM soft body | **3×3** | 3n | yes | indefinite → PSD-proj → SPD | Newton + (P)CG |
| Projective Dynamics | 3×3 | 3n | yes | **SPD, constant** | prefactored Cholesky (or PCG) |
| Backward-Euler deformables | 3×3 | 3n | yes | SPD (proj) | matrix-free (P)CG |
| Fluid pressure Poisson | **1×1** | #cells | yes | **SPD (M-matrix)** | PCG + IC |
| Heat / diffusion (implicit) | 1×1 | #nodes | yes | SPD/PSD | PCG / multigrid |
| Graph Laplacian | 1×1 | #verts | yes | **PSD, singular** | CG |
| Circuit MNA | 1×1 | #nodes | G-block sym; full MNA indefinite | mixed | LU / CG on SPD part |
| Articulated body | **6×6** | 6n | yes | SPD | Featherstone ABA |
| Rigid contact LCP (Delassus `J M⁻¹ Jᵀ`) | 3×3 | #constraints | **sym frictionless; NON-sym w/ Coulomb** | PSD | **matrix-free PGS** |

**Design implication:** the customers for an *assembled* global sparse matrix are **deformables
(3×3 sym SPD)** and **grid/graph PDE solves (1×1 sym SPD/PSD)**. Rigid-body contact stays
matrix-free with tiny local blocks (1×1..6×6) and belongs to a future PGS/impulse layer, not the BSM
assembly path. Non-symmetry's main home is frictional LCP — supported, but not the first target.

Sources: Baraff & Witkin "Large Steps in Cloth Simulation" (SIGGRAPH '98) `A = M − h·∂f/∂v − h²·∂f/∂x`
3n×3n sym SPD; Sifakis & Barbič FEM course (3×3 blocks, PSD projection); Bouaziz et al. Projective
Dynamics (constant SPD, prefactored); Bridson/Greif fluid PCG+MIC0; Featherstone spatial 6×6;
Erleben LCP course; Bullet (Coumans GDC'14: "we don't build an actual 'A' matrix").

---

## 2. Storage format

### 2.1 The type — `fProxyBSM` (namespace `LinearAlgebra.Sparse`)

BSR = block CSR. Uniform block size `BR × BC` for the whole matrix.

```
struct fProxyBSM : IDisposable
{
    int BlockRows;      // number of block-rows        (mb)
    int BlockCols;      // number of block-cols        (nb)
    int BR;             // rows per block
    int BC;             // cols per block
    bool Symmetric;     // upper-block-only storage; requires BR==BC && BlockRows==BlockCols

    // logical scalar dims:  M_Rows = BlockRows*BR,  N_Cols = BlockCols*BC

    // CSR-of-blocks index structure (arena-owned UnsafeLists):
    UnsafeList<int>    RowPtr;    // length BlockRows+1
    UnsafeList<int>    ColInd;    // length nnzb        (block-column of each stored block)
    UnsafeList<fProxy> Values;    // length nnzb*BR*BC  (flat SoA; block k at [k*BR*BC ..])

    Arena* _arenaPtr;             // same ownership pattern as fProxyMxN
}
```

- **Block interior layout: row-major** (`Values[k*BR*BC + r*BC + c]`), matching the library's
  existing row-major dense convention (`fProxyMxN.Data[r*N_Cols+c]`). Documented once; do not mix.
  (cuSPARSE exposes a ROW/COL flag, MKL flips by index-base — we pick one and stick to it.)
- Blocks within a block-row stored in ascending `ColInd`. Enables merges, transpose-SpMV, and a
  binary search for random block access.

### 2.2 Two lifecycle states

1. **Assembly** — a `fProxyBSMBuilder` that accumulates `(blockRow, blockCol, BR×BC block)` triplets
   (COO-of-blocks). Add/remove a node = add/remove triplets. Duplicates summed on compress. This is
   the "sparse matrix is a graph" editable phase.
2. **Compressed** — `builder.ToBSM(arena)` sorts + compresses into BSR. For repeated solves on a
   fixed pattern: keep the BSM, call `ZeroValues()` + `AddToBlock(i,j, …)` (accumulate into the
   existing pattern by lookup) or a returned block view — **no index rebuild**. This is the
   90%-case fast path (assemble structure once, re-stamp values per frame).

Editing the *pattern* after compression → go back through the builder (or an uncompressed-insert mode
à la Eigen, deferred). We do **not** promise cheap in-place pattern mutation of compressed BSR — no
mainstream library does.

### 2.3 Symmetric mode

- Store only blocks with `ColInd >= blockRow` (upper block triangle).
- SpMV does **two updates per stored off-diagonal block**: `y_i += A_ij·x_j` and `y_j += A_ijᵀ·x_i`;
  diagonal block once. Single-threaded → no scatter race → clean ~2× memory & FLOP win.
- Diagonal blocks: for a truly symmetric operator the diagonal block is itself symmetric; store the
  full BR×BC (=BR×BR) block for simplicity in v1 (half-block packing is a later micro-opt).

---

## 3. Operations & the matvec abstraction

The existing `Solvers.conjugateGradient(in fProxyMxN A, …)` calls `Linear_OP.dot(in A, …)` on a
**concrete dense matrix**. To reuse the Krylov solvers for BSM without duplicating them, introduce a
**Burst-friendly static-dispatch operator interface** (generic struct constraint — no managed
delegates, fully inlinable/Burst-compilable):

```
interface IfProxyLinearOperator
{
    int Rows { get; }
    int Cols { get; }
    void Apply (in fProxyN x, ref fProxyN y);   // y = A x
    void ApplyT(in fProxyN x, ref fProxyN y);   // y = Aᵀ x   (for CGLS/LSQR/BiCGSTAB)
}
```

- Thin wrappers implement it: `fProxyBSMOperator` (SpMV over BSR) and `fProxyDenseOperator`
  (wraps `fProxyMxN`). Krylov solvers become generic: `cg<TOp>(in TOp A, …) where TOp : struct,
  IfProxyLinearOperator`. Burst monomorphizes → zero-cost static dispatch, no vtable.
- Existing concrete `conjugateGradient(in fProxyMxN, …)` overloads stay (thin forwarders over the
  dense operator) so nothing downstream breaks.

**SpMV kernels** (`Sparse_OP` / `SpMV`): the block matvec is the single hot kernel. Unroll for
`BR,BC ∈ {1,2,3,4,6}`; general fallback for others. Each block is contiguous → aligned SIMD loads;
this is exactly where the existing register-tile GEMM lessons transfer. `y = Aᵀx` reuses the same
storage (walk blocks, accumulate transposed).

---

## 4. Solver roadmap (dense RHS, matrix-free)

Ranked simplicity-vs-value; each reuses the SpMV above.

1. **CG** (SPD) — 1 matvec + 2 dots + 3 axpy/iter, O(n) memory (3–4 vectors). Warm-startable (seed
   `x₀` from last frame). The default for cloth/FEM/Poisson/Laplacian.
2. **Block-Jacobi preconditioner → PCG** — invert the `BR×BR` diagonal blocks (a batched tiny dense
   inverse — leverages existing GEMM/solve). Cheap to (re)build, big bang-for-buck, ideal fit for
   block-sparse. Point-Jacobi (`diag(A)`) is the `1×1` degenerate case.
3. **MINRES** — symmetric *indefinite* (saddle-point / KKT / PSD-projected-off FEM). Same short
   Lanczos recurrence & O(n) memory as CG, minimizes residual 2-norm (won't break down where CG does).
4. **BiCGSTAB** — non-symmetric (frictional LCP operator, MNA), flat O(n) memory. GMRES(m) as a
   robustness fallback when BiCGSTAB stalls (deferred — O(m·n) memory).
5. **CGLS / LSQR** — rectangular / over-under-determined least squares. Matrix-free via `Apply` +
   `ApplyT`, never forms `AᵀA`. This is the payoff of rectangular blocks.
6. **Block Gauss-Seidel / SOR / (projected) PGS** — the games-physics workhorse. Needs the `BR×BR`
   diagonal-block solve. Graph-colored / red-black variant recovers parallelism later. Truncated
   (≈4–8 sweeps) + warm-started, à la Sequential Impulses / XPBD.

Preconditioners beyond block-Jacobi (SSOR, IC(0)+shift for SPD, ILU(0) for non-sym) are a later,
opt-in tier — only when Jacobi's iteration count hurts *and* the pattern is stable enough to amortize
the build. AMG deferred (setup cost only amortizes on large, topology-stable grids).

---

## 5. Eigensolvers (matvec-only)

1. **Power iteration** — dominant eigenpair; spectral radius (also useful to auto-tune SOR/Chebyshev),
   PageRank-style ranking. Trivial. Add a shifted/deflation variant for the 2nd eigenpair.
2. **LOBPCG** — a few *smallest* SPD eigenpairs via matvec + (the same block-Jacobi) preconditioner;
   block/BLAS-3 friendly (fits the library's GEMM strengths), more stable than Lanczos, no
   reorthogonalization bookkeeping. Targets: **Fiedler vector** (2nd-smallest Laplacian eigenpair →
   spectral mesh/graph partitioning), **lowest vibration modes** (modal analysis for real-time
   deformables), **λ_min** (stability / conditioning).
3. **Lanczos** (symmetric) — optional later, for extremal eigenvalues when no preconditioner exists;
   *must* include at least partial/selective reorthogonalization (naive Lanczos produces ghost
   eigenvalues — Paige).

Inverse iteration / shift-invert deferred: needs a sparse linear *solve* per step (not matvec-only).

---

## 6. Orderings (optional, later)

- **RCM (reverse Cuthill-McKee)** computed **once at setup on the block graph** (each block = one
  supernode). For *iterative* solvers, unpreconditioned CG is permutation-invariant — RCM's value is
  **SpMV cache locality** (banded access pattern) and improving IC/ILU factor stability, not iteration
  count. Linear-time BFS; cheap.
- AMD / nested dissection matter for *direct* factorization (a future sparse-Cholesky path), not the
  iterative core. Defer.

---

## 7. Performance & correctness notes

- **Cache locality:** blocks are inherently local; flat SoA values buffer + contiguous blocks give
  aligned SIMD loads.
- **Block-size specialization (perf pass, NOT hand-unrolling):** the block matvec inner loops have
  trip counts `BR`/`BC` that are *runtime* fields, so Burst/LLVM can't auto-unroll or fully vectorize
  them. Do NOT hand-write per-size copies of the arithmetic. Instead write the block kernel **once**
  as a generic `spMVBlocked<TDim>` where `TDim` is a tiny struct exposing `const int` block dims, and
  dispatch via `switch((BR,BC))` to the instantiations for `{1,2,3,4,5,6}` (+ a runtime-dims generic
  fallback). Burst monomorphizes each instantiation → the loop bounds become literals → LLVM unrolls
  + vectorizes from one source. C#/Burst 1.8 have no `[Unroll]` attribute; the levers are
  compile-time-constant trip counts (via the generic), plus `Unity.Burst.CompilerServices`
  (`Constant.IsConstantExpression`, `Hint.Assume`, `Loop.ExpectVectorized` diagnostic). Phase 1 ships
  the plain general kernel + a `switch` stub; this specialization is a later perf phase.
- **Determinism:** single-threaded fixed reduction order — consistent with the library's
  deterministic-by-construction property. No atomics needed (symmetric SpMV's second write is safe
  single-threaded).
- **Correctness harness:** every sparse op validated against a dense reference (`fProxyMxN`): build a
  random BSM, expand to dense, assert `SpMV == dense matVecDot`, `CG` solution matches dense CG /
  direct solve, symmetric mode matches full storage, `Aᵀ` matches transpose-then-dense. Gallery
  matrices (Laplacian1D/2D, graph Laplacians) give known spectra for the eigensolvers.
- **Codegen:** author as `*.fProxy.cs` templates (float+double) under
  `CodeGen/TemplateSource/Sparse/`; indices are plain `int` (reuse the `Indices` shared-type pattern
  where a cross-type buffer is needed). Arena factory methods on `Arena` for BSM/builder/workspaces.

---

## 8. Phased build plan

- **Phase 1 — Core.** `fProxyBSM` + `fProxyBSMBuilder` (COO→BSR), SpMV + SpMVᵀ (unrolled 1/2/3/4/6 +
  fallback), symmetric mode, dense-reference tests, Arena factories. *No solvers yet.*
- **Phase 2 — SPD solve.** `IfProxyLinearOperator` + BSM/dense operator wrappers; generic CG; block-
  Jacobi preconditioner + PCG; warm-start. Tests on Laplacian/Poisson galleries.
- **Phase 3 — More solvers.** MINRES (sym indefinite), BiCGSTAB (non-sym), CGLS/LSQR (rectangular).
- **Phase 4 — Stationary.** Block Jacobi/Gauss-Seidel/SOR sweeps (+ graph coloring later).
- **Phase 5 — Eigen.** Power iteration, then LOBPCG (reuses block-Jacobi precond).
- **Phase 6 — Optional.** RCM ordering; IC(0)/ILU(0) preconditioners; (future) sparse Cholesky + AMD.

Each phase: templates → `regen.ps1` → `run-tests.ps1`, green suite before the next.

---

## 9. Decisions (LOCKED 2026-07-01)

1. **Block-size model** — ✅ **uniform rectangular `BR×BC`**. Square (`BR==BC`) is the node/simulation
   case; rectangular covers constraint Jacobians + over/under-determined least-squares. Variable
   per-node block sizes remain out of scope.
2. **Symmetric mode** — ✅ **deferred to a later phase**. Phase 1 ships full-storage BSR only;
   symmetric upper-block mode lands after the core + CG are proven (still requires `BR==BC`).
3. **Namespace** — ✅ **`LinearAlgebra.Sparse`**.
4. **First deliverable** — ✅ **vertical slice through CG (Phases 1–2)**: `fProxyBSM` + builder +
   SpMV/SpMVᵀ, then the operator interface + generic CG + block-Jacobi PCG, solving a Laplacian/
   Poisson system end-to-end, validated against the dense reference.

Build order therefore: **Phase 1 → Phase 2** as one deliverable, then Phases 3–6 as before. Symmetric
mode is folded into a later phase (call it Phase 3.5) rather than Phase 1.
```
