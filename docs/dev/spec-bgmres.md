# Mini-spec: `Krylov.bgmres` — block (multi-RHS) restarted GMRES(m)

## 0. Task

Add a new block-Krylov solver `Krylov.bgmres` — restarted block-GMRES(m) for a general (nonsymmetric)
square `A` and `s` simultaneous right-hand sides — to
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.GMRES.fProxy.cs` (new sibling file; see §1 for
why not the existing `Krylov.Block.fProxy.cs`). It parallels scalar `Krylov.gmres` (Arnoldi + Givens
least-squares, restarted) the way `Krylov.bcg`/`Krylov.bcgrq` parallel scalar `Krylov.cg`: ONE shared block
Krylov subspace built from all `s` right-hand sides at once (block Arnoldi via `ApplyBlock`), not `s`
independent scalar GMRES solves.

## 1. Context already read (do not re-derive from scratch)

- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.GMRES.fProxy.cs` — scalar restarted `gmres<TOp,TPre>`
  (lines 32-223). Structure this mirrors: Arnoldi with modified Gram-Schmidt building an orthonormal basis
  `V[0..m]`, an `(m+1) x m` Hessenberg `H`, an incrementally Givens-rotated least-squares RHS `g`, a
  per-step O(1) residual check (`resnorm = |g[j+1]|`, no extra matvec), restart when the inner loop
  exhausts `m` steps without converging, `maxIter` = TOTAL inner steps across restarts. The `M.IsIdentity`
  fold (zt allocated/used only for a real preconditioner) is reused verbatim.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.fProxy.cs` — `Krylov.bcg` (ridge block-CG,
  lines 121-305) and `Krylov.bcgrq` (deflating block-CG via `LQRP`, lines 306-664). Block vectors are
  `fProxyMxN` with **s ROWS x n COLS** (row = RHS/solution vector, length `n = A.Rows`); this convention
  carries over unchanged. Reused **unmodified** from this file (do not duplicate, do not change signatures):
  `BlockCTV` (dst = Cᵀ·V), `BlockAdd` (Y += sign·T), `CopyBlock` (dst = src, rows x cols),
  `BlockApplyPre<TPre>` (row-loop M.Apply), `CountConverged` (per-column ‖R‖² vs threshold),
  `View`/`RowsView`/`RectView` (same-buffer logical reinterpretation of a **leading, about-to-be-fully-
  overwritten flat prefix** — see the aliasing note in §2), `LQRPRank` (numerical rank off `L`'s
  non-increasing |diagonal|, LQRP's own convention). `bgmres` does **not** need `BlockGram`, `BlockSolveSPD`,
  `BlockZplusT`, `CopyMat`, `BlockScatterAddRows`, `LockConvergedRows` (those are bcg/bcgrq-specific:
  SPD-Gram-solve and column-locking machinery `bgmres` has no equivalent of — see §5).
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQRP.fProxy.cs` — `LQRP.decomp(in A, ref L, ref Q, ref P)`
  for `A` (`m x n`, `m <= n`): `P·A = L·Q`, `P` a row permutation, `L` (`m x m`) lower-triangular with
  non-increasing `|diagonal|` (reveals numerical rank), `Q` (`m x n`) with **orthonormal rows**. `A` is not
  modified. Used here (§3) as the block Arnoldi's deflating "thin-QR of the new block": row-major wide
  matrices are exactly LQRP's native shape, no transpose needed.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QR.fProxy.cs` — plain (non-pivoted) tall/square QR, used
  for the small **block Hessenberg least-squares** (§4): `decompInPlace(ref A_to_Q, ref R)` (requires
  `A_to_Q.M_Rows >= A_to_Q.N_Cols`; "Always reports Success — no failure mode"; `A_to_Q` becomes `Q` in
  place) and the multi-RHS `decompSolve(ref Q, ref R, ref B, ref X)` (`B`: `Q.M_Rows x k`, preserved;
  `X`: `Q.N_Cols x k`, output, must not alias `B`; `X = R⁻¹(QᵀB)` via one GEMM + `Blas.triUpper`).
- `Assets/LinearAlgebra/CodeGen/TemplateSource/Pivot/Pivot.cs` — used only as LQRP's own internal pivoting
  scratch (a fresh small `Pivot`, allocated and immediately disposed per LQRP call, exactly bcgrq's own
  usage pattern). `bgmres` never reads a `Pivot`'s values itself (see §4.4's "no un-permutation needed"
  argument) — item 2 of the priority backlog (the Pivot "Arena dependency?" TODO) is **not** touched here.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/Indices/Indices.cs` — a zero-alloc `int` buffer
  (`Indices(int n, Allocator)`, indexer, `Dispose()`). Used here (not in bcg/bcgrq) to hold the per-cycle
  block-width array `w[0..m]` and prefix-offset array `off[0..m+1]` — Burst-safe, no managed `int[]`.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/BlockSolveInfo.cs` — return struct, unmodified.
- `docs/dev/spec-bcgrq.md` — sibling spec for the deflating-QR block-CG; `bgmres` follows the same
  documentation/comment-policy conventions (contracts only in code, rationale in `OP/DEVLOG.md`) and the
  same "buffers allocated once at a fixed max size, narrowed per iteration via `View`/`RowsView`/`RectView`"
  discipline **except** where noted in §2 (the block Hessenberg accumulator needs a real copy, not a view).
- `docs/dev/spec-block-krylov.md` — original block-Krylov spec; §5 item 4 lists block-GMRES as the last of
  four planned block solvers ("Block Arnoldi (orthogonalize s new vectors, deflate), block Hessenberg
  least-squares. Own Temp workspace." — this spec is that item, fully worked out).

### Reference algorithms (cite in the DEVLOG entry once implemented, not in code comments)
- Saad, Y. & Schultz, M.H. (1986). "GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric
  Linear Systems." *SIAM J. Sci. Stat. Comput.* 7(3) — the scalar algorithm this generalizes.
- Simoncini, V. & Gallopoulos, E. (1996). "Convergence properties of block GMRES and matrix polynomials."
  *Linear Algebra and its Applications*, 247, 97-119 — block Arnoldi + block Hessenberg least-squares
  formulation.
- Morgan, R.B. (2005). "Restarted Block-GMRES with Deflation of Eigenvalues." *Applied Numerical
  Mathematics*, 54(2), 222-236 — restarted block-GMRES with basis-rank deflation (the `w[j]`-shrinks
  mechanism in §4.2 below is this paper's deflation, without the eigenvalue-recycling half, which is out of
  scope — see §15).

## 2. Why a sibling file, and the buffer-view aliasing rule

**New file** `OP/Krylov.Block.GMRES.fProxy.cs` (partial `Krylov`), not an addition to
`Krylov.Block.fProxy.cs`: the workspace (§10) is materially larger and structurally different (a persistent
block-Hessenberg accumulator + a per-basis-block `UnsafeList<fProxyMxN>`, vs bcg/bcgrq's handful of `s x n`
/ `s x s` buffers) and shares almost no private helpers beyond the ones listed in §1 — a separate file keeps
`Krylov.Block.fProxy.cs` from growing past a natural "SPD block-CG family" scope. `LQRPRank` stays defined
in `Krylov.Block.fProxy.cs` (it is `private static` inside the same `partial class Krylov` — C# partial
classes share private members across files in the same assembly/namespace, so no duplication, no visibility
change needed).

**Aliasing rule this spec depends on throughout** (state this explicitly to the coder — it is the one
subtlety that breaks the design if missed): `View`/`RowsView`/`RectView` (LOBPCG's/bcgrq's trick: a
value-copy of the `fProxyMxN` struct with `M_Rows`/`N_Cols` overwritten) is safe **only** as (a) the
destination of an operation that fully overwrites the reshaped region (a GEMM output, a `CopyBlock`
destination), or (b) a **row-count-only** prefix view (`RowsView`) of a buffer whose `N_Cols` never changes
from its true allocated stride. It is **not** safe to `RectView` an existing buffer with a **narrower
`N_Cols` than its true stride** and then treat that view as if it were a real sub-matrix of the original
2-D layout for a bulk/stride-blind operation (`Data.CopyFrom`, or passing it as `QR.decomp`'s `in A`, whose
internal `Q.Data.CopyFrom(A.Data)` uses the flat `UnsafeList` length, not `M_Rows*N_Cols`) — doing so
silently copies the wrong elements. Concretely in §4.5 below: the block-Hessenberg accumulator `Hbuf`
(fixed true stride `m*s`) must be copied element-by-element (via `CopyBlock`, whose indexer reads with
`Hbuf`'s own stride and writes with the destination's own stride) into a **separately `RectView`'d** scratch
buffer before that scratch is handed to `QR.decompInPlace`; `Gbuf` (fixed true stride `s`, never reshaped,
only ever `RowsView`'d) needs no such copy.

## 3. Data model and shapes

`n = A.Rows`, `s = B.M_Rows` (fixed RHS count), `m = restart`. Require `s <= n` (LQRP's own `m <= n`
contract, same requirement bcgrq imposes).

### 3.1 Per-cycle block-Krylov basis

A restart cycle builds an **orthonormal block basis** `V[0], V[1], ..., V[k]` (`k <= m`), each `V[i]` a
`w[i] x n` slice of a pre-allocated `s x n` buffer (`w[i] <= s`, **monotonically non-increasing** within one
cycle: `s >= w[0] >= w[1] >= ... >= w[k]` — a block Krylov subspace can only lose rank as it grows, never
regain it within a cycle that never re-orthogonalizes against a discarded direction; a fresh restart resets
`w[0]` from a freshly recomputed residual). `V[0]` comes from a rank-revealing factorization of the cycle's
initial residual block `R0` (not from an Arnoldi step); `V[1..k]` come from the deflating thin-LQ of each
Arnoldi step's post-Gram-Schmidt residual (§4.2).

### 3.2 Block Hessenberg / least-squares accumulator

`w[0..m]` and prefix offsets `off[0..m+1]` (`off[0]=0`, `off[i+1]=off[i]+w[i]`) index two persistent
buffers, both zeroed at the **start of every restart cycle** (cheap — see §10 — and removes any
stale-buffer-from-a-previous-cycle risk when a later cycle's widths are smaller than an earlier one's):

- `Hbuf` (`(m+1)*s x m*s`, fixed true stride `m*s`): block-row `i`, block-column `j` (`0 <= j < k`,
  `0 <= i <= j+1`) holds `H[i,j]` (`w[i] x w[j]`) at absolute offset `(off[i], off[j])`, written via a small
  manual copy helper (§7 `StoreBlockAt`), never via a reshaped view of `Hbuf` itself.
- `Gbuf` (`(m+1)*s x s`, fixed true stride `s`, **never reshaped in the column dimension** — only ever
  `RowsView`'d, so no copy-out is ever needed for it): block-row `0` (rows `[0, w[0])`) holds `G0`
  (`w[0] x s`, `R0` expressed in `V[0]`'s orthonormal basis); block-rows `1..k` are always exactly zero
  (never written — this is deliberate, mirrors scalar `gmres`'s `g[i]=0 for i>0`; see §4.2).

At any point after `k` Arnoldi steps, `totalRows = off[k+1]`, `totalCols = off[k]` (note: **not**
`off[k+1]` — see the boxed note in §4.2, the last built block `V[k]` contributes a Hessenberg **row**-group
but never a **column**-group, exactly matching scalar GMRES's `v_m` / `H`'s `(m+1) x m` shape). `Hbuf`'s
active region is `totalRows x totalCols`; `totalRows >= totalCols` always (since `w[k] >= 0`), satisfying
`QR.decompInPlace`'s tall-or-square requirement.

## 4. The recurrence

### 4.1 Outer restart loop

```
thr[c] = tol^2 * ||B[c,:]||^2, for c in [0, s)         // computed once, original RHS order
total = 0; status = MaxIterations; minActive = s

while total < maxIter and not converged:
    R0[c,:] = B[c,:] - (A X)[c,:], for c in [0, s)      // via A.ApplyBlock(X, ..., s)
    if CountConverged(R0, thr) == s: status = Converged; break     // no basis needed, X unchanged

    zero Hbuf's [0, s) x [0, m*s) region and Gbuf's [0, s) x [0, s) region  (defensive full reset, §3.2)

    LQRP.decomp(R0, L0, Q0, Ppiv0)   // L0, Q0 sized s x s / s x n; Ppiv0 disposed right after
    w[0] = LQRPRank(L0, s, n); minActive = min(minActive, w[0])
    if w[0] == 0: status = Breakdown; break        // R0 nonzero (checked above) yet rank 0 -- defensive,
                                                     // mirrors bcgrq's analogous guard; should not occur
    V[0] := leading w[0] rows of Q0 (LQRP decomposes directly into V[0]'s buffer, see §4.2)
    off[0] = 0; off[1] = w[0]
    G0 := RowsView(Gbuf, w[0]);  Blas.dot(V[0]_view, R0, ref G0, false, true)     // w[0] x s

    k, Y, cycleConverged = InnerArnoldiLoop(...)     // §4.2 -- advances `total`, fills Hbuf/Gbuf,
                                                      // returns the LAST computed least-squares solution Y
    Commit(X, Y, V, w, off, k)                        // §4.3 -- X += combine(Y, V[0..k-1])  (k > 0 always
                                                        // reached here -- see the invariant note in §4.3)

    if cycleConverged: status = Converged; break
    if total >= maxIter: status = MaxIterations; break
    // else: loop back -- fresh restart, R0 recomputed from the just-updated X
```

### 4.2 Inner block-Arnoldi loop (builds `V[1..k]`, `H`'s columns `0..k-1`)

For `j = 0, 1, ..., m-1` while `total < maxIter`:

- **Block matvec (preconditioner fold, identical structure to scalar `gmres`'s inner-step fold).**
  `Vj := RowsView(V[j]_buf, w[j])`. If `M.IsIdentity`: `Wj := RowsView(Wbuf, w[j])`,
  `A.ApplyBlock(Vj, ref Wj, w[j])`. Else: `Ztj := RowsView(Zt, w[j])`,
  `BlockApplyPre(M, Vj, ref Ztj, w[j], n, rowIn, rowOut)` (`Ztj = M⁻¹ Vj`, row-looped — reused unmodified),
  then `A.ApplyBlock(Ztj, ref Wj, w[j])` (`Wj = A M⁻¹ Vj`).
- **Modified block Gram-Schmidt against `V[0..j]`, with ONE unconditional reorthogonalization pass**
  (2 total passes — "MGS2", standard practice for numerical robustness, per the task's requirement to
  specify reorthogonalization explicitly):
  ```
  for pass in {0, 1}:
      for i = 0 .. j:
          Vi := RowsView(V[i]_buf, w[i])
          Hij := RectView(HijBuf, w[i], w[j])                          // w[i] x w[j] scratch
          Blas.dot(Vi, Wj, ref Hij, false, true)                        // Hij = Vi * Wj-transpose
          StoreBlockAt(ref Hbuf, off[i], off[j], Hij, w[i], w[j])       // Hbuf[off[i]+.., off[j]+..] += Hij
          Tij := RowsView(Tbuf, w[j])                                   // w[j] x n scratch
          BlockCTV(Hij, Vi, ref Tij)                                    // Tij = Hij-transpose * Vi
          BlockAdd(ref Wj, Tij, -1)                                     // Wj -= Tij
  ```
  (`StoreBlockAt` accumulates with `+=`; since `Hbuf`'s block `(i,j)` region was just zeroed this cycle
  before any pass wrote to it, pass 0's `+=` is equivalent to `=`, and pass 1 correctly adds the
  reorthogonalization's correction on top.)
- **Deflating thin-LQ of the residual (the "thin-QR of the new block" / basis-rank deflation point).**
  `Ppiv2 := new Pivot(w[j], Temp)`; `Lv := View(Lbuf, w[j])`; `Qout := RowsView(V[j+1]_buf, w[j])` (LQRP
  decomposes directly into `V[j+1]`'s own pre-allocated `s x n` buffer, reshaped to `w[j]` rows for the
  call — no separate `Qfull` scratch is needed since the buffer's own capacity, `s x n`, always covers
  `w[j] x n`); `LQRP.decomp(Wj, Lv, Qout, Ppiv2)`; `Ppiv2.Dispose()`. `w[j+1] := LQRPRank(Lv, w[j], n)`;
  `minActive := min(minActive, w[j+1])`; `off[j+2] := off[j+1] + w[j+1]`.
  If `w[j+1] > 0`: `Vj1 := RowsView(V[j+1]_buf, w[j+1])`; `Hj1j := RectView(HijBuf, w[j+1], w[j])`;
  `Blas.dot(Vj1, Wj, ref Hj1j, false, true)`; `StoreBlockAt(ref Hbuf, off[j+1], off[j], Hj1j, w[j+1], w[j])`.
  (See §4.4 for why this single GEMM, not an un-permutation of `Lv`, is the correct — and exact, not
  approximate — way to obtain `H[j+1,j]`.)
- `total += 1`; `k := j+1`.
- **Least-squares solve + cheap per-column residual check** (§4.5): `totalRows := off[k+1]`,
  `totalCols := off[k]`. Solve, get `Y` (`totalCols x s`) and per-column residual-squared estimates; if all
  `<= thr[c]`: `cycleConverged := true`.
- **Stop this cycle's inner loop** if `cycleConverged` or `w[j+1] == 0` (happy breakdown: the block Krylov
  subspace stopped growing — §4.5 argues the just-computed `Y` is then already exact for the reachable
  subspace, so no further step can help this cycle).

> **Invariant** (state this to the coder, it justifies §4.3's commit loop needing no width-zero guard):
> because the loop breaks the FIRST time `w[j+1] == 0` is discovered (setting `k = j+1` at that same step),
> every `w[i]` for `i` in `[0, k)` used by the commit step is **strictly positive** — `w[k]` itself (the
> "extra" row-group, never used as a search direction) may be zero, but nothing with index `< k` ever is.

### 4.3 Commit: `X += combine(Y, V[0..k-1])`

```
zero Wcombo (s x n)
for i = 0 .. k-1:
    Yi := RectView(YiBuf, w[i], s);  ExtractRowsAt(Y, off[i], w[i], ref Yi)   // Yi = Y[off[i]:off[i]+w[i], :]
    Vi := RowsView(V[i]_buf, w[i])
    Ti := RowsView(Tbuf, s)                                                   // s x n
    BlockCTV(Yi, Vi, ref Ti)                                                  // Ti = Yi-transpose * Vi
    BlockAdd(ref Wcombo, Ti, +1)
if M.IsIdentity: BlockAdd(ref X, Wcombo, +1)
else: BlockApplyPre(M, Wcombo, ref Zcombo, s, n, rowIn, rowOut); BlockAdd(ref X, Zcombo, +1)
```
This runs whenever the inner loop performed at least one step (`k >= 1`) — guaranteed by construction
whenever control reaches this point (the only `k == 0` exit, "R0 already converged," returns before this
step — see §4.1).

### 4.4 Why `H[j+1,j] = V[j+1] · Wj-transpose` is exact (not an approximation) — the reconstruction identity

`LQRP.decomp(Wj, Lv, Qout, Ppiv2)` gives `Ppiv2·Wj = Lv·Qout` with `Qout` (`w[j] x n`) **orthonormal
rows**. `V[j+1]` is defined as the leading `w[j+1]` rows of `Qout` — an orthonormal set spanning (up to the
deflated/negligible part) `Wj`'s row space. For any orthonormal row-set `{v_b}` spanning (a subspace
containing, up to that negligible part) the rows `{w_a}` of a matrix, the standard reconstruction identity
`w_a ~= sum_b (v_b . w_a) v_b` holds with the coefficient matrix `C := V·Wj-transpose` (`C[b,a] = v_b . w_a`)
— i.e. `Wj ~= C-transpose · V`. Setting `V := V[j+1]`, `C := H[j+1,j] = V[j+1] · Wj-transpose` **is** this
reconstruction coefficient, computed by a single GEMM call using the exact same `Blas.dot(..., false, true)`
shape already used for `H[i,j], i <= j` — **no un-permutation of `Lv` via `Ppiv2`'s inverse is needed** (a
permutation-index-based derivation gives the identical result, `H[j+1,j][b,a] = Lv[Ppiv2_inverse[a], b]`,
algebraically — the GEMM route is simpler to specify and implement and is what the coder should use). This
is why `Ppiv0`/`Ppiv2` are disposed immediately after each `LQRP.decomp` call and never otherwise
referenced.

### 4.5 The block Hessenberg least-squares — design decision: periodic dense re-solve, not incremental block-Givens

**This is the high-effort part the coder should budget the most review time for.** A faithful block-GMRES
maintains the least-squares factorization *incrementally* across Arnoldi steps via a sequence of `2w x 2w`
block Givens/Householder rotations (the direct block generalization of scalar `gmres`'s 2x2 Givens pair per
step) — cheap (O(1) amortized) per step, but the block bookkeeping (rotations acting on variable-width row
groups as `w[j]` changes step to step, plus applying each stored rotation to every later column) is
substantial extra machinery with real correctness risk.

**Decision for this implementation: re-run a full dense QR of the accumulated `Hbuf` prefix at every inner
step instead.** This is mathematically identical (same least-squares problem, same optimal `Y`, solved by
QR either way) and is the design already reflected in §4.2's pseudocode:

```
HQ := RectView(HQscratch, totalRows, totalCols); CopyBlock(Hbuf, ref HQ, totalRows, totalCols)   // real copy -- §2
Rls := RectView(Rscratch, totalCols, totalCols)
QR.decompInPlace(ref HQ, ref Rls)                       // HQ becomes Q in place
Gactive := RowsView(Gbuf, totalRows)                    // no copy needed -- Gbuf's N_Cols never changes
Yv := RowsView(Yscratch, totalCols)
QR.decompSolve(ref HQ, ref Rls, ref Gactive, ref Yv)    // Yv = R-inverse(Q-transpose Gactive), current best combo

QtG := RowsView(QtGscratch, totalCols)
Blas.dot(HQ, Gactive, ref QtG, true, false)             // Q-transpose Gactive, totalCols x s
for c in 0 .. s-1:
    gg = sum over r in [0,totalRows) of Gactive[r,c]^2
    qq = sum over r in [0,totalCols) of QtG[r,c]^2
    resid2[c] = max(0, gg - qq)                          // Pythagorean LS-residual identity (thin Q
                                                          // has orthonormal columns: ||b||^2 = ||Q^T b||^2
                                                          // + ||residual||^2), clamped for float round-off
```
`resid2[c] <= thr[c]` for every `c` is the per-column convergence test (§4.2's `cycleConverged`). This
avoids a second, separate matvec-based residual check (mirrors scalar `gmres`'s "no extra matvec" cheap
check, generalized). Cost: `Hbuf`'s active region is at most `(m+1)s x ms` — independent of `n` — so this
re-solve is trivial next to a single matvec on any realistic problem; the redundancy between the `QtG`
computation here and `decompSolve`'s own internal `Q-transpose Gactive` is accepted (both operate on the
same small matrices). **Do not implement incremental block-Givens for this task** — record it as a deferred
optimization in the DEVLOG entry (§13 last item), the same way bcgrq deferred its own "factor once, solve
twice" optimization.

## 5. What `bgmres` deliberately does NOT do (contrast with bcg/bcgrq)

No per-RHS-column **locking**: bcg/bcgrq track a shrinking `sLive`/`Live` (converged columns permanently
drop out of the shared search subspace, mirroring LOBPCG's `numActive` lock loop). `bgmres` has no
equivalent — every column stays "live" through the whole cycle (all `s` columns share the same `Hbuf`/`Gbuf`
least-squares system; a column that individually converged early simply has a near-zero corresponding
column of the residual estimate from then on). This is a legitimate first-cut simplification (task scope,
§15), not an oversight — see §15 for the deferred "deflation of converged systems" enhancement.

## 6. Preconditioner fold

Identical structure/contract to scalar `gmres`'s `M.IsIdentity` fold: `Zt`/`rowIn`/`rowOut`/`Zcombo` are
allocated (and the `BlockApplyPre` branches taken) **only** when `!M.IsIdentity`; under
`fProxyIdentityPreconditioner` every `M`-touching branch constant-folds away (same `TPre`-generic-struct
mechanism `IfProxyPreconditioner.IsIdentity` already documents) and `bgmres<TOp>` (unpreconditioned) must be
**bit-identical** to `bgmres<TOp, fProxyIdentityPreconditioner>` on the same inputs (§12 test 4).

## 7. New private helpers (add to the new file; do not touch the ones reused from §1)

- `static void StoreBlockAt(ref fProxyMxN dst, int rowOff, int colOff, in fProxyMxN src, int rows, int cols)`
  — `dst[rowOff+a, colOff+b] += src[a,b]` for `a < rows, b < cols`. Absolute-index accumulate; safe
  regardless of `dst`'s true stride vs `src`'s (per §2's rule — this is exactly the "manual copy loop"
  escape hatch, not a view).
- `static void ExtractRowsAt(in fProxyMxN src, int rowOff, int rows, ref fProxyMxN dst)` — `dst[a,c] =
  src[rowOff+a, c]` for `a < rows`, all `c < dst.N_Cols` (`== src.N_Cols`, both fixed-width buffers, e.g.
  `Gbuf`/`Yscratch`'s `s`-wide columns — never reshaped in that dimension, so this is a safe absolute-row
  read).
- `static void ZeroPrefix(ref fProxyMxN buf, int rows, int cols)` — `buf[r,c] = 0` for `r < rows, c < cols`.
  Used for the per-cycle `Hbuf`/`Gbuf`/`Wcombo` resets (§4.1).

Comment style: contract only (mirrors §1's reused helpers' existing comment style); the §4.4 reconstruction-
identity argument and the §4.5 design-decision rationale belong in `OP/DEVLOG.md` under a `## Krylov.bgmres`
heading (dated, newest-first, per `CLAUDE.md`), **not** in code comments — a one-line contract comment on
`bgmres` itself (what it solves, that it's restarted block-GMRES, that the operator need not be symmetric)
is enough in code.

## 8. Public API — overload ladder

Mirrors scalar `gmres`'s 8-overload ladder exactly, `fProxyMxN B/X` in place of `fProxyN b/x`; no exposed
scratch-buffer params — `bgmres` owns its whole workspace via `Allocator.Temp`, same as scalar `gmres`, NOT
the zero-alloc caller-owned-scratch style of `bcg`/`bcgrq`:

```csharp
// 1. Generic core.
public static BlockSolveInfo bgmres<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                int restart, int maxIter, fProxy tol)
    where TOp : struct, IfProxyLinearOperator
    where TPre : struct, IfProxyPreconditioner

// 2. Unpreconditioned forwarder (default(fProxyIdentityPreconditioner)).
public static BlockSolveInfo bgmres<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                int restart, int maxIter, fProxy tol)
    where TOp : struct, IfProxyLinearOperator

// 3. Dense, via fProxyDenseOperator.
public static BlockSolveInfo bgmres(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X,
                                int restart, int maxIter, fProxy tol)

// 4. Dense with defaults (restart = min(30, A.M_Rows), maxIter = A.M_Rows, tol = sqrtEps).
public static BlockSolveInfo bgmres(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)

// 5. BSR, via fProxyBSROperator.
public static BlockSolveInfo bgmres(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X,
                                int restart, int maxIter, fProxy tol)

// 6. BSR with defaults.
public static BlockSolveInfo bgmres(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)

// 7. Right-preconditioned BSR with ILU(0).
public static BlockSolveInfo bgmres(in fProxyBSR A, in fProxyILU0 M, in fProxyMxN B, ref fProxyMxN X,
                                int restart, int maxIter, fProxy tol)

// 8. Right-preconditioned BSR with ILU(0), defaults.
public static BlockSolveInfo bgmres(in fProxyBSR A, in fProxyILU0 M, in fProxyMxN B, ref fProxyMxN X)
```

## 9. Argument validation

Mirror scalar `gmres`'s core checks, block-widened, plus the `s <= n` requirement bcgrq also imposes:
`A.Rows == A.Cols` (`ArgumentException`, `"bgmres: A must be square"`); `B.N_Cols == A.Rows`
(`"bgmres: B.N_Cols must equal A.Rows"`); `X.M_Rows == B.M_Rows && X.N_Cols == A.Rows`
(`"bgmres: X must match B"`); `restart >= 1` (`"bgmres: restart must be >= 1"`); `maxIter >= 1`
(`"bgmres: maxIter must be >= 1"`); `B.M_Rows <= A.Rows` (`"bgmres: B.M_Rows (s) must be <= A.Rows"`, LQRP's
own `m <= n` contract).

## 10. Memory footprint and allocation strategy

State this explicitly per the task. All `Allocator.Temp`, allocated once before the restart `while` loop
and reused/overwritten every cycle — mirrors scalar `gmres`'s own "allocate once outside the loop" pattern,
just block-widened:

| buffer | shape | elements | notes |
|---|---|---|---|
| `V` | `(m+1)` entries, each `s x n` | `(m+1) s n` | `UnsafeList<fProxyMxN>`, dominant term — same order as scalar `gmres`'s `O(m n)` `V`, times `s` (expected: it holds `s` independent basis directions per Krylov depth) |
| `Wbuf`, `Tbuf`, `R0`, `Wcombo` | `s x n` each | `4 s n` | Arnoldi-image / MGS-subtract / residual / commit scratch |
| `Zt`, `Zcombo`, `rowIn`, `rowOut` | `s x n`, `s x n`, `n`, `n` | only if `!M.IsIdentity` | preconditioner scratch, identical fold to scalar `gmres`'s `zt` |
| `Hbuf` | `(m+1)s x m s` | `(m+1) m s^2` | the block-Hessenberg accumulator — `O(m^2 s^2)`, no `n` dependence; this is the one term with **no scalar-`gmres` analogue in kind** (scalar's `H` is `O(m^2)` scalars; here each "entry" is an `s x s` block, hence the extra `s^2`) |
| `Gbuf` | `(m+1)s x s` | `(m+1) s^2` | least-squares RHS accumulator |
| `HQscratch`, `Rscratch`, `Yscratch`, `QtGscratch` | `(m+1)s x ms`, `ms x ms`, `ms x s`, `ms x s` | `O(m^2 s^2)` | reused every inner-step check via `RectView`/`RowsView` (§2) |
| `Lbuf`, `HijBuf`, `YiBuf` | `s x s` each | `3 s^2` | small per-block GEMM/LQRP scratch |
| `thr` | length `s` | `s` | |
| `w`, `off` | `Indices(m+1)`, `Indices(m+2)` | `2m+3` | int, not `fProxy` |

For realistic sizes (`m` up to ~30 default, `s` a handful) the `Hbuf`-family terms are at most tens of KB —
negligible next to `V`'s `O(m s n)` for any non-trivial `n`. State this comparison in the DEVLOG entry once
measured; do not hand-wave a number without benchmarking.

## 11. Edge cases

- `B` already within tolerance at a cycle's start (`R0`'s `CountConverged == s` before any `LQRP.decomp`
  call): `status = Converged`, no basis built, `X` unchanged for that check (§4.1). Covers the whole-solve
  case where `X`'s initial (possibly warm-started) guess is already good enough — `total` may be `0`.
- `w[0] == 0` with `R0` nonzero: `status = Breakdown` (defensive; should not occur — `R0` has some column
  with positive norm by the check just above, so its rank is `>= 1`).
- `w[j+1] == 0` for some `j < m-1` (basis-rank exhausted before the residual actually reached tolerance):
  commit the best achievable `Y` for this cycle (§4.3), then continue to the next restart cycle (a fresh
  `R0` from the updated `X` may reveal a richer subspace) — **not** treated as `Breakdown` on its own; only
  `w[0] == 0` at a cycle's very start is (see previous bullet — that is the "genuinely nothing new to try"
  case, since `V[0]`'s LQRP directly factors the fresh residual).
- `s == 1`: degenerates to (numerically) the same computation as scalar `gmres` through the same code path
  (`LQRP.decomp` on a `1 x n` block trivially gives `w[0] = 1` unless `R0 == 0`; `Hbuf`/`Gbuf` become
  `1`-wide-block scalars). No special-casing required; the `s=1` case does not need a dedicated test.
- Warm-started `X` (nonzero on entry): must work — nothing in §4 assumes `X` starts at zero.
- Two (or more) **exactly identical** columns of `B`: `LQRP.decomp` on `R0` must reveal `w[0] < s` on the
  very first cycle (§12 test 5).
- `maxIter` exhausted mid-cycle (`total == maxIter` reached inside the inner loop): the inner loop's `for j`
  condition (`total < maxIter`) stops it; fall through to `Commit` with whatever `k` was reached, then the
  outer loop's `total >= maxIter` check sets `status = MaxIterations`.

## 12. Tests — new file
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BlockGmresTests.fProxy.cs`

Mirror `BlockCGTests.fProxy.cs`'s structure exactly: one `[BurstCompile(CompileSynchronously = true)] IJob`
struct with a `TestType` enum switch, every scenario built and asserted **inside** `Execute()`, each `[Test]`
method doing `new ...Job { Type = ... }.Run()`. `Assert.IsTrue(bool)` / `Assert.AreEqual` only — never the
string-message overload (BC1071 → silent Mono fallback). Build a `BuildDenseNonsymmetric` helper (diagonally
dominant, NOT symmetrized): `A = arena.fProxyRandomMat(dim,dim,-1,1,seed); for d in 0..dim: A[d,d] += dim;`
(GMRES's domain — do not reuse `BlockCGTests`' `M^T M`-symmetrized `BuildDenseSPD`).

1. **`MatchesScalarGmresPerColumn`** — `Krylov.bgmres` on a well-conditioned dense nonsymmetric system
   matches `s` independent scalar `Krylov.gmres` solves to tolerance, and `info.Solved` / `info.converged
   == s`.
2. **`KnownSolutionRecovered`** — known `Xk`, `B = A*Xk` via `fProxyDenseOperator(A).ApplyBlock`, recover
   `Xk` to tolerance.
3. **`RestartCorrectness`** — pick `n` and `restart = m` such that `m < n` and the block subspace cannot
   converge within one cycle (e.g. `n = 40, s = 3, restart = 5`, `maxIter` large enough to allow several
   cycles); assert `info.Solved` **and** `info.iterations > restart` (proves at least one restart actually
   happened, not just that it converged trivially within cycle 1).
4. **`IdentityFoldBitIdentical`** — `bgmres<TOp, fProxyIdentityPreconditioner>` (explicit identity) produces
   **bit-identical** `X`/`iterations`/`status` to the unpreconditioned `bgmres<TOp>` overload on the same
   fixed-seed system (exact double-equality, no tolerance).
5. **`DeflatesRankDeficientRHSBlock`** — force two columns of `B` exactly identical (`s >= 3`); assert
   finite/no-NaN `X`, `info.Solved`, matching columns equal, and `info.minActive < info.rhs` (proves the
   basis-rank deflation in §4.2 actually triggered, mirroring bcgrq's analogous assertion).
6. **`PreconditionedMatchesScalar`** — BSR nonsymmetric system + `fProxyILU0`, matches per-column scalar
   `Krylov.gmres(A, M, ...)` (the preconditioned scalar overload).
7. **`JobSafeThroughRun`** — not a separate scenario, satisfied by construction (every test above already
   runs via `IJob.Run()`); confirm in review only (no extra assertion needed) that `bgmres`'s core never
   reassigns which physical buffer a persistent workspace variable points to mid-solve (no ping-pong /
   `SwapMat`-style buffer-identity hazard — every buffer here is written in place or via `View`/`RowsView`/
   `RectView` of a fixed allocation, so there is nothing analogous to LOBPCG's `RestoreBufferIdentity` need).

All tests use `Consts.fProxySqrtEps`-scale tolerances and the same `Tol()`/`Row`/`DenseToBSR1x1` private
helper style as `BlockCGTests.fProxy.cs` (copy/adapt into the new file, do not import cross-file).

## 13. Implementation checklist (ordered)

1. Create `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.GMRES.fProxy.cs` (new `partial class
   Krylov`).
2. Add the three private helpers (§7).
3. Implement the generic core `bgmres<TOp, TPre>` per §4 exactly (outer restart loop §4.1, inner Arnoldi
   loop §4.2, commit §4.3), reusing the helpers from §1 unmodified.
4. Add the 7 forwarding/convenience overloads per §8.
5. Regenerate (`Tools/regen.ps1`) and confirm float+double both compile clean.
6. Write `BlockGmresTests.fProxy.cs` (§12, all 6 numbered scenarios + the #7 review note).
7. Run the full suite headlessly; confirm the exact line `Result=Passed total=N passed=N failed=0` (never
   pipe through `| tail`).
8. Add a `## Krylov.bgmres` DEVLOG.md entry (per `CLAUDE.md` format: dated, newest-first) capturing: the
   §4.4 reconstruction-identity argument, the §4.5 periodic-re-QR-vs-incremental-block-Givens tradeoff and
   why it was deferred, and the §10 memory-footprint comparison once measured. Do **not** put any of this in
   code comments.

## 14. Acceptance criteria

- `Krylov.bgmres` exists in the new `OP/Krylov.Block.GMRES.fProxy.cs` with the 8-overload ladder of §8,
  generated cleanly for both `float` and `double`.
- No edits to `Krylov.Block.fProxy.cs` beyond what's strictly required to make its existing `private static`
  helpers (`BlockCTV`, `BlockAdd`, `CopyBlock`, `BlockApplyPre`, `CountConverged`, `View`, `RowsView`,
  `RectView`, `LQRPRank`) reachable from the new partial-class file — which, being `private static` members
  of the same `partial class Krylov` in the same namespace/assembly, requires **no signature or
  accessibility change at all** (verify this compiles as-is before considering any visibility edit).
- All 6 numbered tests in `BlockGmresTests.fProxy.cs` (§12) exist and pass, including
  `RestartCorrectness`'s `info.iterations > restart` assertion and `DeflatesRankDeficientRHSBlock`'s
  `info.minActive < info.rhs` assertion.
- `IdentityFoldBitIdentical` passes with **exact** (non-tolerance) equality.
- Full project test suite green: the literal line `Result=Passed total=N passed=N failed=0` from the
  headless test run, `N` including the 6+ new bgmres tests, `failed=0`.
- No edits to `README.md`. No edits to anything under `Assets/LinearAlgebra/Source/` (generated output —
  regenerate instead). No edits to `Pivot/Pivot.cs`, `Pivot/Pivot.Operations.cs`, or `BlockSolveInfo.cs`.

## 15. Out of scope (do not do these in this task)

- Incremental block-Givens/block-Householder least-squares updates (§4.5) — deferred; the periodic dense
  re-QR is the shipped design for this task.
- Per-RHS-column locking / "deflation of converged systems" (§5) — every column stays in the shared
  subspace for the whole cycle; a future enhancement, not this task.
- Eigenvalue recycling across restarts (the other half of Morgan 2005's "deflation of eigenvalues," beyond
  the basis-rank deflation this spec already implements) — not touched.
- Block-MINRES / block-BiCGStab (items 2-3 of `docs/dev/spec-block-krylov.md` §5) — not touched.
- Resolving the `Pivot` "Arena dependency?" TODO (priority-backlog item 2) — `bgmres` only uses `Pivot`
  exactly as it exists today, and only as LQRP's own internal scratch (never reads a `Pivot`'s values, per
  §4.4).
- Any change to `bcg`, `bcgrq`, `BlockSolveInfo`, or any of the reused private helpers' signatures.
- SVD, least squares, optimizers, sparse-matrix work (beyond the existing BSR rungs), View/Slice — unrelated
  to this task.
