# Level-3 (GEMM) Blocking Guide

*Historical document — method names predate the 2026-07 solver-API rework (see
docs/dev/spec-solver-api-rework.md for the mapping).*

How to raise a dense O(n³) kernel from level-2 (gemv / rank-1 / per-row-dot inner loops, memory-
bandwidth bound) to level-3 (GEMM block updates). Written from the QR/LQ compact-WY work (commits
`4f04e76`, `cc087c9`); the durable one-liners also live in `perf-vectorization-lessons.md` point 6.
Read this before blocking Cholesky / LU / SYTRD / bidiag.

## Why blocking wins (and by how much)
A right-looking level-2 factorization re-streams the whole trailing matrix once **per column/reflector**
→ O(n) passes over O(n²) data = O(n³) memory traffic. Blocking `nb` steps into one block update
re-streams it once **per panel** → O(n/nb) passes. The arithmetic is identical; you buy back memory
bandwidth. **Realistic gain here is ~1.3–1.4×, NOT the textbook 2–3×**, because:
- `matMatDot` (our GEMM) tops out ~70 GFLOP/s **untiled** — the block update hits that ceiling but no
  higher. Register-tiling `matMatDot` is the single highest-leverage cross-cutting perf task; it would
  lift every blocked kernel at once. Until then, expect ~1.3×.
- The panel factorization and any reconstruction phase stay level-2; blocking can't GEMM-ify them.
Don't oversell the result. Measure float AND double: the win is bigger for float (double is more
bandwidth-bound and `matMatDot`-double tops out lower).

## Diagnosing whether a kernel is a candidate
- Its O(n³) hot loop is a trailing-matrix update (rank-1 axpy, symmetric rank-2, or a reduction gemv).
- `float == double` in the benchmark ⇒ not SIMD-limited, likely bandwidth-limited ⇒ blocking helps.
- A *reduction*-shaped inner loop (per-row dot, symmetric gemv `p=βAv`) gets the LARGER win: blocking
  turns a loop Burst can't auto-vectorize (loop-carried accumulator under strict FloatMode) into a GEMM.

## Two recipes

### A. Householder reduction (QR, LQ, SYTRD, bidiag) — compact-WY
Batch `nb` reflectors `H_i = I − τ_i v_i v_iᵀ` into a block reflector `I − V T Vᵀ`:
1. **Factor the panel** with the existing unblocked rank-1 code, but restrict each reflector's apply to
   the panel's own columns/rows (add a `colEnd`/`rowEnd` bound). Cheap — the panel is `nb` wide.
2. **Form T** (`formT`, LARFT): `T` is `nb×nb` upper-triangular; `T[i,i]=τ_i` (=1 in this codebase's
   folded convention), off-diagonals from a recurrence over the Gram `VᵀV`. `formT` computes the Gram
   as a unit-stride GEMM-shaped loop (NOT strided per-pair dots — those don't vectorize and cost ~10ms
   at n=1024), then the O(nb³/6) recurrence. Cost is ~nb/n of total → negligible.
3. **Block-apply to the trailing submatrix** as GEMM: `C −= V · (Tᵀ · (Vᵀ C))` (or the folded
   `C −= (C Vᵀ)·(T V)` for a right-apply). Helpers: `wyVtC` (VᵀC), `wyTriTransMul`/`wyTriMul` (Tᵀ/T
   in-place triangular), `wySubVW` (C − VW), and `lqYeqCVt` (CVᵀ for right-apply folds).

**THE DIRECTION LANDMINE (the #1 source of silent corruption):**
| | factorization applies | reconstruction applies |
|---|---|---|
| QR (left-multiply)  | `I − V Tᵀ Vᵀ` → `wyTriTransMul` | `I − V T Vᵀ` → `wyTriMul` |
| LQ (right-multiply) | `I − Vᵀ T V`  → `wyTriMul`       | `I − Vᵀ Tᵀ V` → `wyTriTransMul` |
LQ is the exact opposite of QR because it right-multiplies. Derive it for pb=2 by hand every time.
In-place triangular multiplies must iterate the safe direction (Tᵀ downward, T upward) so no W-row is
overwritten before it's read.

### B. Right-looking factorization (Cholesky POTRF, LU GETRF) — SYRK/GEMM trailing update
No reflectors; the trailing update is a rank-1 (Cholesky) or rank-1 (LU) subtraction:
1. **Factor a `b`-column panel** with the existing unblocked code (Cholesky: the panel's columns; LU:
   panel + partial pivoting restricted to the panel, then apply the row swaps across the full matrix).
2. **Update the trailing block once** as a level-3 op:
   - Cholesky: `A_trail −= L_panel · L_panelᵀ` — a symmetric rank-b update (SYRK) via `matMatDotTransA`.
     Only the lower triangle need be touched; updating the full block is simpler and still correct.
   - LU: `TRSM` to get the panel's U block, then `A_trail −= L_panel · U_panel` — a GEMM (`matMatDot`).
3. Pivoted Cholesky (xPSTRF) does NOT block cleanly — leave `choleskyDecompositionPivot` unblocked.

## Working with strided sub-blocks (row-major)
The trailing block is a column-subrange of a bigger matrix → row stride = full `N`, not the block width.
Two options: (a) write helpers that take an explicit leading dimension (`Cld`) — best perf, what the
`wy*` helpers do; (b) copy the block to contiguous scratch, use `matMatDot`, copy back — safer but the
copies erode ~30-50% of the win. Prefer (a). When V and C live in the SAME matrix at disjoint column
ranges, either copy V to a clean contiguous `Vpanel` buffer (also solves the "stored-R/L entries share
columns with V" masking problem) or be careful that `[NoAlias]` stays truthful (disjoint ranges are OK).

## Non-negotiable landmines checklist
- **Direction (Tᵀ vs T, factor vs reconstruct, QR vs LQ).** See the table. Silent corruption.
- **Masking.** Clean `Vpanel`/`Vfull` to zero above each reflector's diagonal — stored R/L entries
  share those columns and must not leak into V.
- **Buffer sizing at the WIDEST panel** (p0=0). Size `W`-scratch for `nb × N` (reconstruction/first
  panel spans full width). Verify max index `< alloc` by hand.
- **Zero-alloc contract.** Only route the *allocating* overload through the blocked core (it can
  `Temp`-alloc panel/T/W scratch). Leave zero-alloc `(ref ws)` / `(ref u, ref w)` overloads UNBLOCKED
  so their documented no-alloc contract holds. Add a new blocked-workspace overload later if needed.
- **Size gate.** Block only above a crossover (QR `n ≥ 2·BLOCK`; LQ MEASURED `m ≥ 512`). A shared
  float/double proxy template must use the SLOWER type's crossover (double crosses later). Below the
  gate, fall back to the validated unblocked path.
- **Block width is a tuned constant**, method-local `const` (a class-level const collides across the
  float/double partial-class generated files → CS0102). QR_BLOCK=32, LQ_BLOCK=64 were measured; don't
  assume, benchmark 16/32/64.
- **`[NoAlias]`** must be truthful — distinct buffers or provably-disjoint ranges only.
- **codegen:** edit TEMPLATES only (`CodeGen/TemplateSource/**`), never `Source/Generated`. Run
  `Tools/regen-and-test.ps1`. New float/double-only helpers must NOT leak into int/short/long/bool
  generated files (follow the existing `axpy`/`matMatDot` filtering).

## Validation protocol (every blocked kernel)
1. Full suite green via `Tools/regen-and-test.ps1` — the primary correctness gate.
2. **Add permanent tests at NON-block-aligned sizes** (last panel `< BLOCK`) AND above the gate — the
   default suite usually only tests tiny sizes that never reach the blocked path. This is where blocked
   bugs hide (last-panel off-by-ones, stride mistakes).
3. Adversarial code review focused on the direction/masking/sizing landmines.
4. Benchmark A/B (float AND double, small + large N): confirm the win, confirm small N doesn't regress.
   Snapshot the baseline before touching code (`benchmark.ps1` overwrites its output).

## Reusable toolkit (already in `Unsafe_OP`, float/double)
- `wyVtC(V, Vld, C, Cld, rows, pb, cw, W)` — `W = VᵀC` (block reflector, left-apply half 1).
- `wySubVW(V, Vld, C, Cld, rows, pb, cw, W)` — `C −= V·W` (half 2). Also serves LQ's `C −= Y·V`.
- `wyTriTransMul(T, pb, W, cw)` / `wyTriMul(T, pb, W, cw)` — in-place `W := Tᵀ·W` / `T·W`.
- `lqYeqCVt(C, Cld, Vt, cn, rows, pb, Y)` — `Y = C·Vᵀ` (right-apply fold; needs Vt = transpose of V).
- `formT(Vp, Vld, rows, pb, T, tcol, G)` — LARFT compact-WY T (currently duplicated in QR and LQ; when
  a 3rd caller appears, promote to a shared `internal` helper).
- `matMatDot` / `matMatDotTransA` — the GEMM / `AᵀB` primitives (Cholesky SYRK, LU GEMM use these). A
  `matMatDotTransB` (`ABᵀ`) is still missing — add it for pseudoInverse/lowRankApprox.
