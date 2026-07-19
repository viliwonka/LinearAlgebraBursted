# Mini-spec: `Krylov.bcgrq` — Block-CG with reliable/deflating QR (LQ) updates

## 0. Task

Add a new, standalone block-Krylov solver `Krylov.bcgrq` to
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.fProxy.cs`, alongside (not replacing) the
existing ridge-regularized block-CG (`Krylov.cg` block overloads). `bcgrq` replaces the ridge-regularized
`s x s` Gram solves with a row-pivoted rank-revealing LQ (`LQRP`) factorization of the (preconditioned)
residual block every iteration, so near-dependent RHS directions are **dropped** (deflated) rather than
patched with a diagonal ridge. Ships with comparison tests (accuracy + iteration count vs ridge `cg` on
ill-conditioned / near-parallel-RHS systems) and a benchmark row.

## 1. Context already read (do not re-derive from scratch)

- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.fProxy.cs` — existing ridge block-CG. Block
  vectors are `fProxyMxN` with **s ROWS x n COLS** (row j = RHS/solution vector j). Private helpers used
  below: `BlockGram`, `BlockCTV`, `BlockAdd`, `BlockApplyPre`, `CopyBlock`, `BlockSolveSPD`,
  `CountConverged`. `BlockZplusT` and `CopyMat` are **not** used by `bcgrq` (noted so the coder doesn't
  go looking for a use).
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/BlockSolveInfo.cs` — return struct, already has
  `minActive` with doc text that explicitly covers **both** "converged" and "linearly dependent" causes —
  `bcgrq` is what actually exercises the "linearly dependent" half; ridge `cg` always reports
  `minActive == rhs`.
- `docs/dev/spec-block-krylov.md` — original block-Krylov spec. Its §2 flagged QRCP/`OrthonormalizeBlock`-
  style deflation as future work for block-CG; `bcgrq` is that follow-up, using `LQRP` (row-native, no
  transpose) instead of LOBPCG's SVQB (this is a linear **solve**, not an eigenproblem — no
  Rayleigh-Ritz, no B-inner-product, no SVQB).
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQRP.fProxy.cs` — row-pivoted rank-revealing LQ.
  `LQRP.decomp(in A, ref L, ref Q, ref P)`: for `A` (`m x n`, `m <= n`), produces `P*A = L*Q` where `P` is
  a row permutation (`Pivot`, size `m`), `L` is `m x m` lower-triangular with **non-increasing `|diagonal|`**
  (reveals numerical rank), `Q` is `m x n` with **orthonormal rows** (`Q*Q^T = I_m`). `A` is not modified
  (the non-destructive overload — used here because the raw residual is still needed after
  factorization). Rank is read off `L`'s diagonal by the caller (the same convention
  `LQRP.solveInPlace` uses internally): `tol = relTol * |L[0,0]|`, `relTol = max(m,n) * Consts.fProxyZeroThreshold`,
  `rank = count of leading i with |L[i,i]| > tol`. Use the **plain, non-cache overload**
  (`LQRP.decomp(in A, ref L, ref Q, ref P)`) — it already does its own internal `Allocator.Temp` scratch,
  and building a custom `fProxyLQRPCache`-shaped view is a micro-optimization explicitly **out of scope**
  (note it as a DEVLOG follow-up once benchmarked, do not implement it now).
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.fProxy.cs` — mined for two mechanics, **not**
  for its eigensolver machinery:
  - `View(in fProxyMxN buf, int m)` / `RowsView(in fProxyMxN buf, int rows)` (LOBPCG.fProxy.cs:931,944):
    same-buffer, smaller-shaped logical reinterpretation (a value-copy of the `fProxyMxN` struct with
    `M_Rows`/`N_Cols` overwritten — not a new allocation, not a strided sub-block of a larger stride, a
    **contiguous-prefix reinterpretation**). This is the mechanism that lets `bcgrq` allocate every
    scratch buffer once at the max size `s` (or `s x s`) and use a narrower logical view each iteration
    as the active width `sa`/`sLive` shrinks or grows back — this is what answers the "variable-stride
    scratch" concern raised for this task: **no dedicated variable-stride struct is needed**, just
    `View`/`RowsView`/a new `RectView` (§7) over fixed-max-size buffers, exactly LOBPCG's own pattern.
  - The lock/swap-to-back pattern (LOBPCG.fProxy.cs:253-331, backward `for (i = numActive-1; i >= 0; i--)`
    scan, swap a satisfied row to the current last slot, `numActive--`): reused for **convergence locking**
    (§4) — NOT for the rank-deflation (that is handled entirely inside the width `sa`, no row swapping —
    see §4).
  - **`bcgrq` does not touch Rayleigh-Ritz, SVQB, B-inner-products, or any LOBPCG eigensolver code.** It
    is a standalone linear solver.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/CHO.fProxy.cs` — `CHO.decompSolve(ref L, ref B_to_X)`
  requires only `B_to_X.M_Rows == L.M_Rows`; `B_to_X.N_Cols` is unconstrained. `BlockSolveSPD` (in
  `Krylov.Block.fProxy.cs`) already relies on exactly this generality (its `RHS_to_X` is `s x s` today
  but the helper never assumes square) — **reused unmodified** for `bcgrq`'s **rectangular** `sa x sLive`
  (alpha) and `sa x saNew` (beta) solves. No change to `BlockSolveSPD`'s signature or body.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/Pivot/Pivot.cs` — `Pivot(int size, Allocator)`, fixed size
  at construction, `this[i]` getter, `Swap(i,j)`, `Reset()`. Used two ways below: (a) a **persistent**
  `Pivot Live` (size `s`, allocated once, never resized) tracking "physical row slot -> original RHS
  index" for the locking mechanism (§4); (b) a **fresh per-iteration** `Pivot` (size `sLive`, allocated
  and disposed inside each LQRP call) for LQRP's own internal row-pivoting scratch — `Pivot` has no
  resizable/view mechanism (unlike `fProxyMxN`), so unlike every other small scratch buffer here (which
  is allocated once at max size `s` and reused via `View`), this one is intentionally a fresh small
  `Allocator.Temp` alloc every iteration; this mirrors how `LQRP.decomp`'s own allocating overload
  already does the same thing per call and is not a new allocation pattern for this codebase.
  (Item 2 of the priority backlog, "finish the Pivot struct" / arena-dependency TODO, is **not** touched
  by this task — `bcgrq` only *uses* `Pivot` as it exists today.)

## 2. Data model and shapes

- `s` = number of original RHS (rows of `B`/`X`, fixed for the whole solve). `n` = `A.Rows`. Require
  `s <= n` (an `ArgumentException`, matching `LQRP`'s own `m <= n` contract — block-CG's use case is
  always `s` small relative to `n`).
- `sLive` (int, starts at `s`, **monotonically non-increasing**): number of original RHS columns not yet
  converged. Shrinks only via the convergence-lock step (§4.1).
- `sa` (int, recomputed **fresh every iteration**, **not monotonic** — can go up or down): the numerical
  row-rank of the live (preconditioned) residual block, `sa <= sLive` always. This is the "reliable QR"
  deflation width (§4.2).
- `Live` (`Pivot`, size `s`, persistent): `Live[i]` for `i` in `[0, sLive)` is the **original** row index
  currently sitting at physical row `i` of the internal `R` buffer. Locking a row swaps it to the current
  last live slot and shrinks `sLive` (§4.1) — this permutes `R`'s rows but **`X`'s rows are never
  permuted** (see §4.1 for why and how the X update stays correctly indexed).

## 3. The recurrence

This is a from-first-principles, internally-consistent derivation in the paper's spirit (breakdown-free
block CG via a rank-revealing factorization of the residual in place of forming/inverting a possibly
rank-deficient Gram matrix — O'Leary, "A generalized conjugate gradient algorithm for solving a class of
quadratic programming problems", 1980, block-CG recurrence; Dubrulle, "Retooling the method of block
conjugate gradients", ETNA vol. 14, 2001, the "reliable" QR-based residual factorization this is named
after; Ji & Li, "A breakdown-free block conjugate gradient method", BIT Numer. Math., 2017, deflation via
rank-revealing factorization). Equation/step numbers below (S1, S2, ...) are this spec's own — the exact
equation numbers in those papers were not reproduced verbatim (paraphrase the cited technique, do not
claim to match their notation line-for-line). **Treat the derivation below as authoritative; it has been
checked for shape-consistency and for reducing to standard block-CG (up to an orthonormal change of basis,
which does not change the Galerkin-projection iterates in exact arithmetic) whenever `sa == sLive == s`
throughout.**

### 3.1 State carried between iterations

| name | shape | role |
|---|---|---|
| `X` (public ref, caller/Arena-owned) | `s x n` | solution, **original row order always**, warm-startable |
| `R` (public ref) | `s x n` (max), live prefix `sLive x n` | full residual, rows **permuted** via `Live` |
| `P` (public ref) | `s x n` (max), live prefix `saSearch x n` | current A-conjugate search block |
| `AP` (public ref) | `s x n` (max), live prefix `saSearch x n` | `A * P` (search block's A-image) |
| `Pa` (public ref) | `s x n` (max) | LQRP's `Q` output; leading `sa` rows are this iteration's fresh orthonormal search directions |
| `Z` (public ref, only if `!M.IsIdentity`) | `s x n` (max) | preconditioner scratch, `M^-1 * R_live` |
| `saSearch` (int) | -- | row-count of `P`/`AP`/`PQ` "as of the last update" |
| `PQ` (internal Temp) | `s x s` (max), live `saSearch x saSearch` | `P^T A P`, SPD, carried across the alpha/beta pair in one iteration |

### 3.2 Per-iteration steps (shapes explicit; `View`/`RowsView`/`RectView` = §7 helpers)

Precondition entering the loop body: `P`, `AP`, `PQ` already hold this iteration's `P_search`/`AP_search`/
`PQ_search` at width `saSearch` (established by §3.3 setup, or by the previous iteration's S11-S13).

- **S1. Gather the live residual (zero-copy).** `Rlive = RowsView(R, sLive)`.
- **S2. Alpha RHS.** `alpha = RectView(alphaBuf, saSearch, sLive)`;
  `Blas.dot(in Psearch, in Rlive, ref alpha, false, true)` -- i.e. `alpha := P_search * R_live^T`
  (`saSearch x sLive`). This is literally block-CG's own `alpha = (P^T A P)^-1 (P^T R)` formula with `P`
  narrowed to `saSearch` rows and `R` narrowed to the `sLive` still-live rows -- no substitution, no
  reinterpretation needed.
- **S3. Solve for alpha.** `work = View(workBuf, saSearch)`;
  `ok = BlockSolveSPD(in PQ, ref alpha, ref work, saSearch)`. `PQ` is SPD by construction (`A` SPD);
  `BlockSolveSPD`'s own ridge-retry is the **last-resort guard** here (should essentially never trigger --
  see §5). `!ok` -> `status = Breakdown`, `iterations = k`, done.
- **S4. Update X (scatter -- X keeps original row order).**
  `T = RowsView(Tbuf, sLive)`; `BlockCTV(in alpha, in Psearch, ref T)` (`T := alpha^T * P_search`,
  `sLive x n`); `BlockScatterAddRows(ref X, in T, in Live, sLive, (fProxy)1)` (§7) --
  `X[Live[i], :] += T[i, :]` for `i` in `[0, sLive)`. **Never reorder X's rows.**
- **S5. Update R (in place, still permuted -- plain `BlockAdd` applies).**
  `T2 = RowsView(Tbuf, sLive)` (reuse -- S4's `T` is fully consumed);
  `BlockCTV(in alpha, in AP_search, ref T2)` (`T2 := alpha^T * AP_search`);
  `Rlive = RowsView(R, sLive)`; `BlockAdd(ref Rlive, in T2, (fProxy)(-1))`.
- **S6. Lock converged rows (§4.1), shrinking `sLive`.** If `sLive == 0` after locking:
  `status = Converged`, `iterations = k+1`, done.
- **S7. Fresh (preconditioned) live residual.** Identity fold: `Zlive = RowsView(R, sLive)`; else
  `Rlive2 = RowsView(R, sLive)`, `Zpre = RowsView(Z, sLive)`,
  `BlockApplyPre(in M, in Rlive2, ref Zpre, sLive, n, ref rowIn, ref rowOut)`, `Zlive = Zpre`.
- **S8. Rank-revealing factorization (§4.2).** Fresh `Pivot Ppiv = new Pivot(sLive, Allocator.Temp)`;
  `Lv = View(Lbuf, sLive)`; `Qfull = RowsView(Pa, sLive)`;
  `LQRP.decomp(in Zlive, ref Lv, ref Qfull, ref Ppiv)`; `Ppiv.Dispose()`. Read `saNew` off `Lv`'s
  diagonal (§1's rank convention, `m = sLive`). `minActive = min(minActive, saNew)`. `PaActive =
  RowsView(Pa, saNew)` (the leading `saNew` rows of the **same** `Pa` buffer -- valid regardless of the
  `sLive` used for the decomp call, since `RowsView` is just a smaller prefix of the same storage).
  `saNew == 0` (with `sLive > 0`, i.e. every live row's own norm is above `tol` yet the *rank* collapsed
  to zero -- should not happen in practice; see §4.2) -> `status = Breakdown`, `iterations = k+1`, done.
- **S9. Beta RHS (A-conjugacy of `P_new` against `P_search`; derivation in §5).**
  `beta = RectView(betaBuf, saSearch, saNew)`;
  `Blas.dot(in AP_search, in PaActive, ref beta, false, true)` -- `beta := AP_search * Pa_new^T`
  (`saSearch x saNew`). **Uses the SAME `AP_search`/`PQ` from S1-S3** (this iteration's `A*P_search`,
  computed once, reused for both the alpha RHS's `PQ` factorization and this beta RHS -- no extra
  matvec).
- **S10. Solve for beta.** `work2 = View(workBuf, saSearch)`;
  `ok2 = BlockSolveSPD(in PQ, ref beta, ref work2, saSearch)`. `!ok2` -> `Breakdown`, `iterations = k+1`,
  done.
- **S11. Form `P_new` -- READ `P_search` BEFORE overwriting `P`'s storage (aliasing hazard, see boxed
  warning below).**
  1. `Tb = RowsView(Tbuf, saNew)`; `BlockCTV(in beta, in Psearch, ref Tb)` -- `Tb := beta^T * P_search_OLD`
     (`saNew x n`), computed **first**, into the separate `Tbuf`, while `Psearch = RowsView(P, saSearch)`
     still holds the OLD data.
  2. `Pnew = RowsView(P, saNew)`; `CopyBlock(in PaActive, ref Pnew, saNew, n)` -- **now** safe to overwrite
     `P`'s storage (`Pnew` is `Pa_new`).
  3. `BlockAdd(ref Pnew, in Tb, (fProxy)(-1))` -- `Pnew -= Tb`, i.e. `P_new := Pa_new - beta^T * P_search`.

  > **Aliasing warning (write this up explicitly, do not let the coder reorder these):** `RowsView(P,
  > saSearch)` and `RowsView(P, saNew)` are views of the **same** backing buffer `P`. If `saNew != saSearch`
  > these views' row ranges only partly overlap. Step 1 **must** fully finish reading `Psearch` (and write
  > its result to the separate `Tbuf`, not into `P`) before step 2 starts writing into `P`. This is the
  > exact class of bug LOBPCG's `RestoreBufferIdentity` note (LOBPCG.fProxy.cs:963) warns about for
  > ping-ponged buffers -- `bcgrq` avoids needing a ping-pong/`SwapMat` at all (P/AP/Pa are never
  > *reassigned* to a different physical buffer, only overwritten in place, so there is no
  > IJob-struct-copy buffer-identity hazard to begin with) **provided** this read-before-write order is
  > respected.
- **S12. `A * P_new`.** `APnew = RowsView(AP, saNew)`; `A.ApplyBlock(in Pnew, ref APnew, saNew)` -- the
  **one** matvec this iteration (same per-iteration matvec count as ridge `cg`).
- **S13. `P_new`'s own Gram.** `PQnew = View(PQbuf, saNew)`; `BlockGram(in Pnew, in APnew, ref PQnew,
  saNew)`.
- **S14. Roll state.** `saSearch := saNew`. (`P`, `AP`, `PQ` buffers already hold the new data at the new
  width via the views written above; `sLive`/`Live` were already updated at S6.) Loop to S1, or if
  `k+1 == maxIter`: `status = MaxIterations`, `iterations = maxIter`, done.

### 3.3 Setup (before the loop; establishes iteration 0's `P_search`/`AP_search`/`PQ_search`)

1. Validate arguments (§6).
2. `thr = fProxyN(s)`; `thr[j] = tol*tol * sum_c B[j,c]^2` for all `j` in `[0, s)` -- **original** order,
   computed once from `B` before any permutation ever happens.
3. `Live = new Pivot(s, Allocator.Temp)` (identity: `Live[i] == i`). `sLive = s`. `minActive = s`.
4. `R := B - A*X`: `A.ApplyBlock(in X, ref AP, s)` (reuse `AP` as scratch -- mirrors ridge `cg`'s own reuse
   of `Q` for this); `R[i,c] = B[i,c] - AP[i,c]` for all `s` rows.
5. Lock pass over the full `s` rows (§4.1, `Live` still identity so `Live[i] == i`). If `sLive == 0`:
   `status = Converged`, `iterations = 0`, done (skip everything else).
6. S7-equivalent: `Zlive` from `R`'s current leading `sLive` rows (identity fold as in S7).
7. S8-equivalent: `LQRP.decomp` on `Zlive` -> `sa` (`Pa`'s leading `sa` rows = `PaActive`).
   `minActive = min(minActive, sa)`. `sa == 0` -> `Breakdown`, `iterations = 0`, done.
8. `saSearch := sa`. `Psearch = RowsView(P, saSearch)`; `CopyBlock(in PaActive, ref Psearch, saSearch, n)`
   (`P_search := Pa` -- no `P_old` to conjugate against on the first iteration, so no beta/subtraction
   here, just a copy).
9. `APsearch = RowsView(AP, saSearch)`; `A.ApplyBlock(in Psearch, ref APsearch, saSearch)`.
10. `PQ = View(PQbuf, saSearch)`; `BlockGram(in Psearch, in APsearch, ref PQ, saSearch)`.
11. Enter the loop at S1 with `k = 0`.

### 3.4 Cleanup (every exit path funnels here)

Recompute the residual **fresh from the final `X`** (do not try to unpermute the internal working `R` --
simpler, and doubles as an exit-time sanity check in the spirit of the existing
`KrylovVerifyAtExitTests.fProxy.cs` convention):

```
Rfinal = RowsView(AP, s);                      // reuse AP at full width s
A.ApplyBlock(in X, ref Rfinal, s);
for i,c: Rfinal[i,c] = B[i,c] - Rfinal[i,c];
CountConverged(in Rfinal, in thr, s, n, out maxRnorm)  -> converged
```
`BlockSolveInfo { rhs = s, converged = <from CountConverged>, iterations = <set at the exit point>,
maxRnorm = <from CountConverged>, minActive = minActive, status = <set at the exit point> }`.

Dispose (in reverse-allocation order, matching ridge `cg`'s cleanup style): `thr`, `Live`, `alphaBuf`,
`betaBuf`, `PQbuf`, `Lbuf`, `workBuf`, `Tbuf`, and `rowIn`/`rowOut` if allocated.

## 4. Deflation policy -- two independent, unified-by-`minActive` mechanisms

### 4.1 Convergence locking (drives `sLive`)

After the R update (S5, or the initial `R` at setup), scan `i` from `sLive-1` down to `0`
(**exact mirror of LOBPCG's lock loop**, LOBPCG.fProxy.cs:253/308/331 -- backward scan so a swap-in from
the current last slot is still re-tested):
```
orig = Live[i];
rr = sum_c R[i,c]^2;
if (rr <= thr[orig]) {
    last = sLive - 1;
    if (i != last) { Swap.Rows(ref R, i, last); Live.Swap(i, last); }
    sLive--;
}
```
A locked row's `X` value is **frozen** (never updated again -- S4 only ever touches `X[Live[i],:]` for `i`
in the *current* `[0, sLive)`, so once a row is locked it drops out of every subsequent S4). Its `R` value
stays at whatever it was at lock time (correct -- it already satisfied `tol`). `converged` (final,
reported) is *not* derived from `sLive`; it comes from the fresh cleanup recompute (§3.4) -- the two
should agree in a correct implementation but the report uses the independently-recomputed one.

### 4.2 Rank deflation (drives `sa`, independent of `sLive`)

Every iteration, `LQRP.decomp` factors the **entire current live residual block** (all `sLive` rows, not
just some subset) and `sa <= sLive` is read off `L`'s diagonal (§1). **`sa < sLive` does NOT remove any
column from `sLive`/`Live`/X-updating** -- every live column keeps receiving an X/R update every iteration
via `alpha`'s `sLive`-wide column dimension (S2/S4/S5), regardless of `sa`. Rank deflation only narrows
the *dimension of the search subspace* (`P`/`AP`/`PQ`'s row count) used to compute that shared update --
this is the "already-committed partial solutions in X keep converging through the shared subspace; do not
zero them" behavior called out in `docs/dev/spec-block-krylov.md` §2, now concretely realized: a
column that is *linearly dependent* on other live columns' residual directions still gets updated, because
`alpha`'s RHS (S2) is computed against the **full** `R_live` (all `sLive` columns), not against some
already-reduced subset.

`sa` is recomputed from scratch every iteration and is **not monotonic** -- if the live residual block's
conditioning improves (e.g. a previously near-dependent direction separates out as other columns
converge), `sa` can go back up on a later iteration. `minActive = min` over every `sa` (and the initial
setup's `sa`) seen during the whole solve -- this is what `BlockSolveInfo.minActive` reports.

`sa == 0` while `sLive > 0` is treated as `Breakdown` (defensive guard only -- should not occur: the
`sLive` rows all have per-column norm `> tol` by construction of the lock test, so the largest live row
alone forces `L[0,0] != 0` and rank `>= 1` whenever `sLive > 0`; if it ever fires it indicates every live
row is numerically a linear combination that cancels to below the *rank* tolerance while each individually
stays above the separate *convergence* tolerance -- a pathological, not expected, case).

## 5. Why the recurrence is correct (for the coder's confidence, and to justify §3's exact formulas)

- **Alpha (S2/S3)** is literally block-CG's own formula, `alpha = (P^T A P)^-1 (P^T R)`, with `P` and `R`
  independently narrowed to `saSearch` and `sLive` rows respectively -- no reinterpretation needed, it
  typechecks and matches classical block-CG exactly when `saSearch == sLive == s`.
- **Beta (S9/S10/S11)** is derived from the **A-conjugacy requirement** directly (not by pattern-matching
  the classical ratio-of-Gram-matrices formula, which does not typecheck once `saSearch != saNew`): demand
  `P_new * A * P_search^T == 0` (`saNew x saSearch` zero block) for the ansatz
  `P_new = Pa_new - beta^T * P_search`. Substituting:
  ```
  (Pa_new - beta^T P_search) A P_search^T = 0
  Pa_new A P_search^T = beta^T (P_search A P_search^T) = beta^T PQ_search
  beta^T = (Pa_new A P_search^T) PQ_search^-1
  beta   = PQ_search^-1 (P_search A Pa_new^T) = PQ_search^-1 (AP_search Pa_new^T)     [PQ_search symmetric, A symmetric]
  ```
  which is exactly S9/S10 (`PQ_search * beta = AP_search * Pa_new^T`, solved via `BlockSolveSPD`, then
  `P_new = Pa_new - beta^T P_search` at S11). This is the standard "conjugate Gram-Schmidt" form of the CG
  direction update (mathematically equivalent to the classical ratio-of-Gram-matrices recursion in exact
  arithmetic, by the standard CG/Krylov-subspace orthogonality theory -- the ratio formula is itself only a
  cheap 2-term shortcut for this same A-orthogonalization). It is well-defined for **any** `saSearch`/
  `saNew` pair, including a width change between consecutive iterations, with **no special-casing** --
  this is why no restart-on-deflation logic is needed anywhere in §3.
- **Reliability.** The only matrix ever factored/inverted is `PQ_search = P_search^T A P_search`
  (`saSearch x saSearch`, SPD since `A` is SPD) -- used for *both* alpha and beta in the same iteration
  (computed once at S13/setup-S10, consumed at S3 and S10). `LQRP`'s rank-revealing factorization is what
  keeps `Pa`/the residual well-conditioned in the first place (§4.2); `BlockSolveSPD`'s ridge-retry is
  reserved purely as a **last-resort guard** on `PQ_search` itself (which is not directly protected by the
  LQ step -- it reflects `P_search`'s own accumulated conjugate history, the same generic CG numerical-
  stability concern ridge `cg` already guards against, just on a `saSearch x saSearch` matrix that is
  never larger, and typically much smaller, than ridge `cg`'s full `s x s`).
- **Basis invariance.** In the undeflated case (`sa == sLive == s` every iteration, no locking, no rank
  loss), `Pa_k` is a full orthonormal basis for the same row space as classical block-CG's
  `Z_k = M^-1 R_k` (they differ by an invertible `s x s` change of basis coming out of the LQ
  factorization). Block-CG's iterates are a Galerkin projection onto the block-Krylov subspace, which is
  basis-independent -- so `bcgrq` reduces to the same `X_k`/`R_k` sequence as classical (undeflated) block
  CG in exact arithmetic in this limit, up to floating-point round-off. This is the sense in which
  `bcgrq` is a drop-in, more-reliable alternative rather than a different algorithm.

## 6. Argument validation (mirror ridge `cg`'s block-CG core exactly, plus one addition)

Same checks as `cg<TOp,TPre>`'s core (`A.Rows == A.Cols`; `B.M_Rows == s`, `B.N_Cols == n`; `X`/`R`/`P`/
`AP`/`Pa` all `s x n`; `Z` `s x n` only required if `!M.IsIdentity`; `maxIter >= 1`), **plus**:
`s <= n` (`ArgumentException`, message e.g. `"bcgrq: B.M_Rows (s) must be <= A.Rows (n)"` -- required by
`LQRP.decomp`'s own `m <= n` contract).

## 7. New private helpers to add to `Krylov.Block.fProxy.cs`

- `static fProxyMxN View(in fProxyMxN buf, int m)` and `static fProxyMxN RowsView(in fProxyMxN buf, int
  rows)` -- copy LOBPCG.fProxy.cs:931-949 verbatim (same doc-comment content/spirit: "same-buffer,
  smaller-shaped logical view ... not a new allocation"). These are private to `Krylov` (a different
  partial class than `LOBPCG`), so they must be **added here**, not referenced cross-class.
- `static fProxyMxN RectView(in fProxyMxN buf, int rows, int cols)` -- the same trick generalized to an
  independent row/col count (`v = buf; v.M_Rows = rows; v.N_Cols = cols; return v;`), valid whenever
  `rows*cols <= buf`'s total element capacity. Needed because `alpha`/`beta` are rectangular
  (`saSearch x sLive`, `saSearch x saNew`), carved out of `s x s`-capacity buffers.
- `static void BlockScatterAddRows(ref fProxyMxN Yfull, in fProxyMxN Tlive, in Pivot Live, int sLive,
  fProxy sign)` -- `Yfull[Live[i], c] += sign * Tlive[i, c]` for `i` in `[0, sLive)`, all `c` in
  `[0, n)`. This is the one genuinely new arithmetic helper (§3.2 S4); everything else reuses `BlockGram`
  /`BlockCTV`/`BlockAdd`/`CopyBlock`/`BlockApplyPre`/`BlockSolveSPD`/`CountConverged` unmodified.

Comment style for all of the above: contract only (what shape it produces / what it destroys / what it
requires), matching the project's comment policy. Anything explaining *why* (the LQ-vs-Gram tradeoff, the
aliasing-order requirement, the basis-invariance argument, benchmark numbers once measured) goes in
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/DEVLOG.md` under a `## Krylov.bcgrq` heading, not in code
comments -- see `CLAUDE.md`'s comment policy. In particular: do **not** put the §5 derivation in code
comments; a one-line contract comment on `bcgrq` itself (what it solves, what it destroys/requires, that
it deflates via LQRP) is enough; the rest belongs in DEVLOG.md.

## 8. Public API -- overload ladder (mirrors `cg`'s ladder structure; one extra required scratch buffer)

```csharp
// 1. Generic core.
public static BlockSolveInfo bcgrq<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                ref fProxyMxN R, ref fProxyMxN P, ref fProxyMxN AP, ref fProxyMxN Pa,
                                ref fProxyMxN Z, int maxIter, fProxy tol)
    where TOp : struct, IfProxyLinearOperator
    where TPre : struct, IfProxyPreconditioner

// 2. Unpreconditioned forwarder (default(fProxyIdentityPreconditioner), default Z -- never dereferenced
//    under the IsIdentity compile-time fold, exactly like cg<TOp>).
public static BlockSolveInfo bcgrq<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                ref fProxyMxN R, ref fProxyMxN P, ref fProxyMxN AP, ref fProxyMxN Pa,
                                int maxIter, fProxy tol)
    where TOp : struct, IfProxyLinearOperator

// 3. Dense SPD, arena-allocating (R/P/AP/Pa via B.fProxyTempMat(s, n, true)).
public static BlockSolveInfo bcgrq(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)

// 4. Dense SPD, default maxIter (A.M_Rows) / tol (Consts.fProxySqrtEps).
public static BlockSolveInfo bcgrq(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)

// 5. Dense SPD, preconditioned, arena-allocating (also allocates Z).
public static BlockSolveInfo bcgrq<TPre>(in fProxyMxN A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                int maxIter, fProxy tol)
    where TPre : struct, IfProxyPreconditioner

// 6. BSR SPD, arena-allocating.
public static BlockSolveInfo bcgrq(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)

// 7. BSR SPD, default maxIter/tol.
public static BlockSolveInfo bcgrq(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)

// 8. BSR SPD, preconditioned, arena-allocating.
public static BlockSolveInfo bcgrq<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                int maxIter, fProxy tol)
    where TPre : struct, IfProxyPreconditioner
```

The one deliberate deviation from an exact parameter-list mirror of `cg` is the extra required `ref
fProxyMxN Pa` buffer (§3.1) -- `bcgrq` genuinely needs one more persistent `s x n` buffer than ridge `cg`
does. Naming (`R`, `P`, `AP` in place of `cg`'s `R`, `P`, `Q`, `Z`) otherwise follows `cg`'s exact
ordering/style; `Q` is renamed `AP` here purely for local clarity (it is always literally `A * P`) -- note
this rename explicitly so it isn't mistaken for a divergence in convention.

## 9. Edge cases to handle explicitly

- `sLive == 0` at setup or after any lock pass -> `Converged`.
- `sa == 0` (setup or S8) with `sLive > 0` -> `Breakdown` (§4.2).
- `BlockSolveSPD` failure at S3 or S10 -> `Breakdown`.
- `s == 1`: degenerates to ordinary (scalar) CG through the same code path -- `LQRP.decomp` on a `1 x n`
  block trivially gives `L = [+-||row||]`, `sa = 1`. No special-casing required; add a smoke test if cheap
  but not mandatory.
- Two (or more) **exactly** identical RHS columns: `LQRP.decomp` must reveal `sa < sLive` on the very
  first iteration where those columns are still both live (existing `RankDeficientBlockDeflates`-style
  scenario, §10).
- Warm-started `X` (nonzero on entry): must work -- nothing in §3 assumes `X` starts at zero.
- `maxIter` exhausted with some columns still live: `status = MaxIterations`, `converged < s`, `X`'s
  locked columns are exact, live columns hold the best iterate reached.

## 10. Tests -- new file `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BlockCGrQTests.fProxy.cs`

Mirror `BlockCGTests.fProxy.cs`'s structure exactly: one `[BurstCompile(CompileSynchronously = true)]
IJob` struct with a `TestType` enum switch, every scenario built and asserted **inside** `Execute()`, each
`[Test]` method doing `new ...Job { Type = ... }.Run()`. Use `Assert.IsTrue(bool)` /
`Assert.AreEqual` only -- **never** the string-message overload (BC1071 -> silent Mono fallback).

Required test cases:

1. **`MatchesScalarCgPerColumn`** -- `Krylov.bcgrq` on a well-conditioned dense SPD system (same
   `BuildDenseSPD` helper style as `BlockCGTests`) matches `s` independent scalar `Krylov.cg` solves to
   `tol`-scaled tolerance, and `info.Solved` / `info.converged == s`.
2. **`KnownSolutionRecovered`** -- known `Xk`, `B = A*Xk` via `ApplyBlock`, recover `Xk` to tolerance.
3. **`RankDeficientBlockDeflatesAndReportsMinActive`** -- two RHS columns forced exactly identical (as
   `BlockCGTests.RankDeficientBlockDeflates`); assert finite/no-NaN, `info.Solved`, matching columns equal,
   **and additionally** `info.minActive < info.rhs` (the concrete difference from ridge `cg`, which always
   reports `minActive == rhs` -- this is the one new assertion this test needs beyond the existing ridge
   version).
4. **`PreconditionedMatchesScalar`** -- BSR + block-Jacobi preconditioner, matches per-column scalar `pcg`
   (`Krylov.cg(in A, in M, in bj, ...)`), mirroring `BlockCGTests.PreconditionedMatchesScalar`.
5. **`IdentityFoldBitIdentical`** -- `bcgrq<TOp, fProxyIdentityPreconditioner>` (explicit identity) produces
   **bit-identical** `X`/`iterations`/`status` to the unpreconditioned `bcgrq<TOp>` overload on the same
   fixed-seed system (exact double-equality on every element, no tolerance) -- mirrors the project's
   existing `MergedCgIdentityMatchesPlainCg`-style test.
6. **`IllConditionedSPDNeverWorseThanRidge`** -- build an ill-conditioned SPD system (use
   `arena.fProxyHilbert(n)` directly, or `M^T M` from a random `M` with a deliberately stretched singular
   spectrum -- e.g. scale `M`'s columns by geometrically spaced factors spanning several decades before
   forming `M^T M`), known solution `Xk` (`B = A*Xk`), solved with **both** `Krylov.cg` (ridge) and
   `Krylov.bcgrq` under the same `maxIter`/`tol` budget. Assert:
   - both report `Solved` (or, if the budget is deliberately tight, compare whichever metrics apply to
     both -- do not assert `Solved` if the case is intentionally too hard for either, see the benchmark for
     the "how hard" tuning knob instead);
   - `bcgrq`'s worst per-column residual (`maxRnorm`) is `<=` ridge `cg`'s (allow a small tolerance-scaled
     slack, e.g. `<= ridgeMaxRnorm * (1 + 1e-6)`, to avoid flaking on a genuine tie);
   - `bcgrq`'s worst per-column forward error `max_j ||X[j,:] - Xk[j,:]||` is `<=` ridge `cg`'s forward
     error (same slack);
   - `bcgrq.iterations <= ridge.iterations * <generous factor, e.g. 2>` (bcgrq pays extra per-iteration
     cost for the LQ factorization; the ask is "not dramatically worse iteration count," not strictly
     fewer iterations -- the accuracy assertions above are the primary claim).
7. **`NearParallelRHSNeverWorseThanRidge`** -- same comparison as #6, but the ill-conditioning source is
   the **RHS block** instead of `A`: build a well-conditioned SPD `A`, then make column 1 of `B` a tiny
   perturbation of column 0 (e.g. `B[1,:] = B[0,:] + eps_scale * randomRow` with `eps_scale` a few orders
   of magnitude above machine epsilon but small enough that the two columns are numerically near-parallel)
   for `s >= 3`. Same three assertions as #6 (`maxRnorm`, forward error, iteration-count sanity).
8. **`JobSafeThroughRun`** -- not a separate scenario, satisfied by construction (every test above already
   runs via `IJob.Run()`); no extra test needed, but confirm in review that `bcgrq`'s core never reassigns
   which physical buffer a `ref fProxyMxN` parameter (`P`, `AP`, `Pa`, `R`) points to (§3.2 S11's boxed
   warning) -- if that invariant holds there is no `RestoreBufferIdentity`-style hazard to test for.

All tests use `Consts.fProxySqrtEps`-scale tolerances and the same `Tol()`/`BuildDenseSPD`/`Row`/
`DenseToBSR1x1` private helper style as `BlockCGTests.fProxy.cs` (copy/adapt, do not import cross-file).

## 11. Benchmark -- extend `BlockCGSparseBenchmark`

Files: `Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/BlockCGSparseBenchmark.fProxy.cs` (generated
per-dtype half) and `Assets/LinearAlgebra/Benchmarks/BlockCGSparseBenchmark.cs` (hand-written harness
half). Keep it **sparse** (BSR 2D Poisson via `arena.fProxyLaplacian2D(grid, grid)`) -- that's the real
block-CG use case; do not add a dense variant.

- Add a `BlockCgrQSparseJobFProxy : IJob` struct mirroring `BlockCgSparseJobFProxy` exactly (same `A`, `B`,
  `X` fields, plus `R, P, AP, Pa` in place of `R, P, Q`; same `Indices Out` convention -- `Out[0] =
  info.iterations`, `Out[1] = info.minActive`), calling `Krylov.bcgrq(new fProxyBSROperator(in A), in B,
  ref X, ref R, ref P, ref AP, ref Pa, K, Tol)`.
- In `BenchFProxy(grid, s)`: allocate `R, P, AP, Pa` (`s x n` each, `arena.fProxyMat`), run the new job via
  `Bench.Time`, append a `"bcgrq"` row using the **same** `fmt`/column layout as the existing `"block-CG"`
  row (`med(ms)`, `min(ms)`, `iters`, `minAct`) so the two rows are directly diff-able in the report.
- Update the hand-written `BlockCGSparseBenchmark.cs`'s header comment to mention the new `bcgrq` row
  (one factual sentence, no benchmark numbers/verdicts -- those go in DEVLOG.md once measured) and keep the
  same `Grids`/`Ss` sweep so the two solvers are compared at identical problem sizes.
- Do **not** add a dense bcgrq row, and do not change the existing `spMM`/`spMVx s` matvec-only probe rows.

## 12. Implementation checklist (ordered)

1. Add `View`/`RowsView`/`RectView`/`BlockScatterAddRows` private helpers to `Krylov.Block.fProxy.cs`
   (§7).
2. Implement the generic core `bcgrq<TOp, TPre>` per §3 exactly (setup §3.3, loop §3.2, cleanup §3.4).
3. Add the 7 forwarding/convenience overloads per §8.
4. Regenerate (`Tools/regen.ps1`) and confirm float+double both compile clean.
5. Write `BlockCGrQTests.fProxy.cs` (§10, all 7 scenarios + the #8 review note).
6. Extend `BlockCGSparseBenchmark` (§11).
7. Run the full suite headlessly; confirm the exact line `Result=Passed total=N passed=N failed=0` (never
   pipe through `| tail`).
8. Add a `## Krylov.bcgrq` DEVLOG.md entry (per `CLAUDE.md` format: dated, newest-first) once benchmark
   numbers exist, capturing: the LQRP-per-iteration cost tradeoff observed, whether the "factor once /
   solve twice" CHO micro-optimization (§1, explicitly deferred) looks worth doing, and any benchmark
   comparison numbers vs ridge `cg`. Do **not** put any of this in code comments.

## 13. Acceptance criteria

- `Krylov.bcgrq` exists in `OP/Krylov.Block.fProxy.cs` with the 8-overload ladder of §8, generated cleanly
  for both `float` and `double`.
- Existing ridge `Krylov.cg` block overloads are **unmodified** (byte-identical file region, or at most a
  whitespace-only diff from adjacent additions).
- `BlockSolveSPD`, `BlockGram`, `BlockCTV`, `BlockAdd`, `CopyBlock`, `BlockApplyPre`, `CountConverged` are
  reused unmodified (same signatures, same bodies) -- no signature changes to any existing helper.
- All 7 tests in `BlockCGrQTests.fProxy.cs` (§10) exist and pass, including the two ill-conditioned
  comparison tests (#6, #7) with their `maxRnorm`/forward-error/iteration-count assertions against ridge
  `cg` on the **same** system/budget.
- `RankDeficientBlockDeflatesAndReportsMinActive` asserts `info.minActive < info.rhs` (an assertion ridge
  `cg`'s own equivalent test does not and cannot make).
- `IdentityFoldBitIdentical` passes with **exact** (non-tolerance) equality.
- Full project test suite green: the literal line `Result=Passed total=N passed=N failed=0` from the
  headless test run, `N` including the 7+ new bcgrq tests, `failed=0`.
- `BlockCGSparseBenchmark` produces a `"bcgrq"` row alongside `"block-CG"`/`"scalar x s"` at every
  `(grid, s)` combination in the existing sweep, both dtypes.
- No edits to `README.md`. No edits to anything under `Assets/LinearAlgebra/Source/` (generated output --
  regenerate instead). No edits to `Pivot/Pivot.cs` or `Pivot/Pivot.Operations.cs` (item 2 of the backlog,
  out of scope here -- `bcgrq` only uses `Pivot`'s existing public surface).

## 14. Out of scope (do not do these in this task)

- The `fProxyLQRPCache`-based zero-Temp-alloc LQRP call (§1) -- noted as a future DEVLOG follow-up only.
- The "factor `PQ_search` once via `CHO.decompInPlace`, solve alpha and beta from the same factor" micro-
  optimization (§5) -- `BlockSolveSPD` is called twice per iteration (once for alpha, once for beta),
  deliberately simple; revisit only if benchmarks show it matters.
- Any change to the ridge `Krylov.cg` block overloads, `BlockSolveInfo`, or any of the reused private
  helpers' signatures.
- Block-MINRES / block-BiCGStab / block-GMRES (items 2-4 of `docs/dev/spec-block-krylov.md` §5) -- not
  touched.
- Resolving the `Pivot` "Arena dependency?" TODO (priority-backlog item 2) -- `bcgrq` uses `Pivot` exactly
  as it exists today.
- Any dense (non-BSR) benchmark row for `bcgrq`.
- SVD, least squares, optimizers, sparse-matrix work, View/Slice -- unrelated to this task.
