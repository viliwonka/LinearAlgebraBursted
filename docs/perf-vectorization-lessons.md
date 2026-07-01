# Burst Vectorization & Perf Lessons (LinearAlgebraBursted)

Hard-won rules from vectorizing the dense kernels. Terse on purpose.

## Diagnosing
- **`float == double` in a benchmark ⇒ NOT SIMD-vectorized.** When vectorized, float runs ~1.5–2× double (float4 vs double2/4). This ratio is the single best diagnostic.
- **Measure in Burst, not Mono.** Run work inside a `[BurstCompile] IJob` via `.Run()`; a plain managed call mis-measures ~10×. Use `CompileSynchronously = true`; warmup + median over N runs; include large N (cache effects show only there).
- **GEMM is the ceiling reference** (~70 GFLOP/s float on this machine). Compare a kernel's GFLOP/s to GEMM to gauge remaining headroom.
- **Bench BOTH float and double; the ratio attributes the win.** A unit-stride rewrite can give a big wall-clock gain (3–6×) yet leave float/double pinned at ~1.3× — that gain was *cache*, not SIMD. Two common reasons SIMD still won't fire on an otherwise-contiguous loop: a reduction (dot) stays scalar under strict `FloatMode`, and two offset pointers into the *same* buffer (e.g. rows `i` and `i+1` of one matrix) can't be proven non-aliasing. Cache-first is still the right order — just don't claim "vectorized" from a wall-clock drop alone.

## Why a kernel won't vectorize
- **Burst vectorizes loops along their iteration axis — not hand-unrolled bodies.** Make the unit-stride axis the *inner loop variable*.
- **Row-major `Data[r*N+c]`:** walking a row = unit-stride (good); walking a column = stride N (cache- and SIMD-hostile). Restructure column-strided kernels to sweep rows.
- The 2D indexer `[r,c]` is **NOT** the blocker — its bounds check is `#if`'d out in release and it inlines to a pointer deref.
- Without an aliasing guarantee Burst assumes pointers may overlap and won't vectorize. `[NoAlias]` is what unlocks it.

## The fixes (in order of leverage)
1. **axpy beats dot.** `y[i] += a*x[i]` is independent across i → full SIMD. A dot/reduction has a loop-carried accumulator that Burst can't reassociate under strict `FloatMode` → stays scalar. Prefer **rank-update (axpy)** formulations over **inner-product (dot)** ones. (Right-looking LU/Cholesky, Householder rank-2 update.)
2. **Route hot loops through the existing `[NoAlias]` raw-pointer `Unsafe_OP` kernels** (`axpy`, `vecDotRange`, `addSquares`, …) — the same path GEMM uses. This is what makes Burst emit SIMD.
3. **If a dot is unavoidable, use multiple independent accumulators** (source-level reassociation). Recovers ILP (overlapped FMA latency) ≈ ~2×, but that's ILP, *not* SIMD width — the real win is a rank-update rewrite. **Sweet spot is narrow, measure it**: LQ's row-Householder right-apply (`M -= (Mv)vᵀ`, a per-row dot — see point 5) went from a naive single-accumulator scalar loop to **4** accumulators for a ~3× win, but **8** accumulators on the *same* loop *regressed* to ~1.7× slower than 4 (register pressure from 8 live accumulator variables overwhelmed whatever the auto-vectorizer/scheduler could do with them). Don't extrapolate "more accumulators = more ILP" — try a couple of widths and keep the fastest.
4. **Cache vs SIMD are separate, stacking wins.** Register-blocking / row-reorder fixes *cache*; it does not vectorize. Do both.
5. **Left-multiply (`uᵀM`, sum of scaled rows) is axpy-friendly in row-major storage; right-multiply (`Mv`, a per-row reduction) is not, and there's no free restructure that fixes it.** `uᵀM`'s shared accumulator is built via *repeated axpy calls* (each covers the full row segment, no reduction) — that's why QR's/Bidiag's left-apply reflector code vectorizes so well. `Mv`'s per-row scalar is fundamentally a reduction over that row; walking down a column instead (the other order that would avoid a reduction) reintroduces the row-major stride-N cache/SIMD penalty this whole doc is about. When an algorithm's natural formulation needs `Mv` (LQ's row-Householder right-apply, direct — not transpose-to-QR), accept the per-row reduction and fix it locally (point 3's multi-accumulator dot), or reconsider whether transposing into the `uᵀM`-friendly direction (paying an O(mn) transpose) is actually cheaper end-to-end — it was competitive here even after the accumulator fix (see LQ's benchmark history in `benchmark-tallwide.txt`: the direct form wins at small/medium N, transpose-to-QR was ~10% faster at N=1024 float before this fix narrowed it to near-parity).

## Algorithm choice beats micro-opt
- **Rotation-based methods (Jacobi / Givens SVD & eigen) are column-oriented** (strided rotations) and resist SIMD. Switch to **Householder reductions** (tridiagonalization / bidiagonalization): they pack the O(n³) work into gemv + rank-2 axpy updates that vectorize, leaving only O(n²) iterative cleanup. (~75× for symmetric eigenvalues vs cyclic Jacobi.)
- **Reuse a fast primitive.** Singular values = positive eigenvalues of the augmented `[[0,A],[Aᵀ,0]]` (Jordan–Wielandt) → reuses the fast symmetric eigensolver and keeps κ(A), **not** κ(A)² (never form AᵀA for singular values).

## Numerical care
- **Preserve bitwise identity when you can:** `(-a)*x == -(a*x)` and `y + (-(a*x)) == y - a*x` exactly in IEEE, so an axpy rewrite is bitwise-identical *if accumulation order is preserved*.
- **Reassociation (multi-accumulator dot, dot-then-subtract, right-looking) is rounding-only, NOT bitwise.** Validate with tolerance tests; state which in the commit.
- `!(x > 0)` is true for NaN — use it for positive-definite / non-finite guards (reject before `sqrt`).

## Workflow
- **A/B every change:** `git stash` the kernel → regen → benchmark (baseline) → restore → regen → benchmark (after). Untracked benchmark files survive path-checkouts.
- **Validate against a trusted oracle:** cross-check a new method (e.g. Householder eigen) against the existing one (Jacobi) on random + known matrices; add adversarial code-review.
- **Scratch:** thread a workspace `ref` param for true zero-alloc hot paths (keep tested contracts); otherwise one `Allocator.Temp` bump per call is fine (O(n) « O(n³)). Never allocate per inner iteration.
- **Codegen bootstrap deadlock:** a test/benchmark referencing a not-yet-generated symbol blocks codegen (project won't compile → codegen can't run). Break it: remove the reference, regen, re-add.
