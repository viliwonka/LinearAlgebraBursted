# Mini-spec: `Krylov.bminres` — block MINRES for symmetric (possibly indefinite) systems

## 0. Task

Add a new block-Krylov solver `Krylov.bminres` to
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.fProxy.cs` (same file as the existing
block-CG core, whether it is still named `cg` or has already been renamed to `bcg` by the time this is
implemented -- this task does not depend on that rename and does not touch it either way). `bminres` is a
TRUE block method (one shared block-Krylov subspace built via block Lanczos, s x s block coefficients,
`ApplyBlock` once per iteration) for a **symmetric, possibly INDEFINITE** operator `A` -- it must NOT assume
SPD, unlike block-CG. It parallels the existing scalar `Krylov.minres<TOp,TPre>`
(`OP/Krylov.fProxy.cs:526`) the same way block-CG parallels scalar `cg`.

## 1. Context already read (do not re-derive from scratch)

- `OP/Krylov.fProxy.cs:526-791` -- scalar `minres<TOp,TPre>` and its forwarders. This is the literal
  template being generalized: Lanczos three-term recurrence, `IsIdentity` compile-time fold for the
  preconditioned vs plain path, the coupled two-term Givens/QR bookkeeping (`cs,sn,dbar,epsln,phibar`),
  the search-direction recurrence (`w = (v - oldeps.w1 - delta.w2)/gamma; x += phi.w`), and the
  verify-at-exit pattern used under preconditioning. Every block quantity below is named to mirror this
  file's scalar names capitalized (`Alfa`<->`alfa`, `Beta`<->`beta`, `Dbar`<->`dbar`, `Epsln`<->`epsln`,
  `Gamma`<->`gamma`, `Phibar`<->`phibar`, `W`/`W1`/`W2`<->`w`/`w1`/`w2`).
- `OP/Krylov.Block.fProxy.cs` -- existing block-CG (`cg`/`bcg`) core and its private helpers, reused
  UNMODIFIED here: `BlockGram`, `BlockCTV`, `BlockAdd`, `CopyBlock`, `CopyMat`, `BlockApplyPre`,
  `CountConverged`. `BlockZplusT`/`BlockSolveSPD` are **not** reused (`bminres` is symmetric-indefinite,
  not SPD -- its own small-matrix solves need a general, not SPD-only, factorization; see SS6). Block
  vectors are `fProxyMxN` with **s ROWS x n COLS** (row = one block-Lanczos vector / one RHS column,
  matching `ApplyBlock`).
- `OP/BlockSolveInfo.cs` -- return struct (`rhs`, `converged`, `iterations`, `maxRnorm`, `minActive`,
  `status`). `bminres` v1 (this task) keeps a **fixed block width `s`** throughout (see SS2) -- `minActive`
  is therefore always reported as `s` (mirrors block-CG's own ridge-first-cut behavior, per its DEVLOG:
  "Fixed block width s throughout (no active-width bookkeeping)... minActive = s").
- `docs/dev/spec-block-krylov.md` SS5 item 2: "block-MINRES (symmetric indefinite). Parallels
  `minres<TOp,TPre>`. Block Lanczos + banded QR; deflation on the s x s Lanczos off-diagonal. Higher effort
  (banded Givens bookkeeping)." -- this task IS that item.
- `docs/dev/spec-bcgrq.md` SS2/SS7 (already-written sibling spec, not yet necessarily implemented): its
  block deflation policy (rank-reveal, drop dependent columns) and its `View`/`RowsView` "same-buffer,
  smaller-shaped logical view" idiom (copied verbatim from `LOBPCG.fProxy.cs:931-949`). **`bminres` v1 does
  NOT need `View`/`RowsView`/`RectView`** -- its block width is fixed at `s` for the whole solve (SS2), so
  every block buffer is used at its full allocated shape every iteration; no variable-width views are
  needed. If `bcgrq` has already added `View`/`RowsView` to this file by the time `bminres` is
  implemented, do not duplicate them; `bminres` simply does not call them.
- `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BlockCGTests.fProxy.cs` -- the test-file
  structure to mirror exactly (one `[BurstCompile(CompileSynchronously = true)] IJob` with a `TestType`
  enum switch, `BuildDenseSPD`/`Row`/`DenseToBSR1x1`/`Tol()` private helpers copied and adapted -- `bminres`
  needs a `BuildDenseSymIndefinite` helper instead of `BuildDenseSPD`, see SS10).
- `OP/CHO.fProxy.cs:274` `CHO.decompSolve(ref L, ref B_to_X)` -- solves `L.L^T.X = B` in place; `B_to_X`
  must have `M_Rows == L.M_Rows`, `N_Cols` unconstrained (each **column** of `B_to_X` is an independent
  RHS vector, ordinary matrix-inverse-multiply semantics: `B_to_X := (L.L^T)^-1.B_to_X`).
- `OP/LU.fProxy.cs:703` `LU.solveInPlace(ref fProxyMxN A_to_LU, ref Pivot P, ref fProxyMxN B_to_X)` --
  general (non-symmetric) square solve, partial pivoting, `B_to_X` is `n rows x k cols` (`n` = matrix
  size, `k` = number of RHS **columns**); same column-RHS convention as `CHO.decompSolve`; returns
  `DirectSolveStatus.Singular` (inside a `DirectSolveInfo`) without solving if singular -- **do not** treat
  a `Singular` return as an exception, treat it as `Breakdown` (SS8).
- `OP/QR.fProxy.cs:518` `QR.decomp(in A, ref Q, ref R)` -- `A` is `m x n`, `m >= n` (tall/square); `Q` is
  **thin** (same shape as `A`, `m x n`, orthonormal COLUMNS); `R` is `n x n` upper-triangular. Always reports
  `Success` (no failure mode -- a zero column falls back to a sign-convention default, see
  `genHouseholder`, `QR.fProxy.cs:30`). Used twice per iteration to build a 2s x 2s orthogonal completion
  (SS4.2) -- no new QR machinery needs to be written, both calls route through this existing primitive.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/Consts.cs` -- `fProxyEpsilon` (float `1.1920929e-7`),
  `fProxyZeroThreshold` (float `1e-6`), `fProxySqrtEps` (float `3.4526698e-4`); doubled analogues exist for
  the `double` half of codegen. Used for the ridge-retry scale and the Qperp rank check (SS4.2, SS6).

## 2. Scope decision for this v1 (read before designing anything else)

Block-CG shipped with **fixed block width `s` + ridge-regularized Gram solves** as its first cut, deferring
true rank-revealing column-dropping deflation to a separate follow-up (`bcgrq`, spec'd but not necessarily
implemented yet). `bminres` v1 makes the **same scope decision, for the same reason**: true deflation of
the block-Lanczos recurrence (shrinking width mid-solve when the residual block loses rank) requires
variable-width Lanczos vectors, a variable-size banded QR, and reworking the coupled two-term recurrence's
shapes every time the width changes -- a second, harder version of exactly the complexity `bcgrq` is already
carrying for block-CG. Building both the fixed-width recurrence AND variable-width deflation for a
brand-new algorithm in one pass is not a one-session unit of work and duplicates risk unnecessarily.

**Decision: `bminres` v1 keeps block width `s` fixed for the entire solve.** A rank-deficient block (e.g.
two identical RHS columns, or a block Lanczos vector candidate that is nearly linearly dependent on what's
already in the subspace) is handled defensively by a **ridge-regularized block-Lanczos normalization**
(SS4.1's `BlockNormalize`, structurally the same ridge-retry technique `BlockSolveSPD` already uses for
block-CG) so the solve never NaNs or throws -- it does **not** shrink the working width or report
`minActive < rhs`. `minActive` is always `s`. True column-dropping block-MINRES deflation (mirroring
`bcgrq`'s relationship to block-CG) is explicitly **out of scope** here (SS13) -- name it e.g. `bminresrq` as
a future follow-up in the DEVLOG entry this task writes (SS11 step 8).

## 3. Data model and shapes

- `s` = block width = number of simultaneous RHS (rows of `B`/`X`), **fixed for the whole solve**. `n` =
  `A.Rows`. Require `s <= n` (mirrors `bcgrq`'s own `s <= n` requirement, needed because a block Lanczos
  vector block cannot have more orthonormal rows than the ambient dimension) -- `ArgumentException` if
  violated.
- Every **block-Lanczos-indexed** quantity below (`Alfa`, `Beta`, `Dbar`, `Epsln`, `Gamma`, `Omega`) is an
  `s x s` (or `2s x 2s`) `Allocator.Temp` scratch matrix, re-allocated (or reused) fresh each iteration --
  same "many small Temp `s x s` buffers" idiom `bcg`'s core already uses for `PQ`/`RZ`/`RZnew`/`coef`/`work`.
- Every **block-vector** quantity (`X`, `B`, `Vprev`, `Vcur`, `Wk`, `W`, `W1`, `W2`, `Z`) is `s x n`,
  caller-provided via `ref fProxyMxN` (arena/job-owned, matching `bcg`'s `R`/`P`/`Q`/`Z` scratch contract) --
  no variable-width views needed (SS2).
- `Phibar`/`Phi` are `s x s` but with a **different semantic pairing than `Alfa`/`Beta`**: rows index the
  current block-Lanczos direction, columns index the **original RHS column** (`B`'s/`X`'s row index).
  Concretely, `Phibar` starts as `Transpose(Beta[0])` where `Beta[0]` comes from normalizing the initial
  residual block `R0 = B - A.X0` (SS4.3) -- see the derivation note in SS4.4 for why the transpose is needed
  (it falls out of `R0`'s natural row-major LQ-style factorization `R0 = Beta[0].V[1]`, while the
  projected least-squares RHS needs the column-major-equivalent `R0^T = V[1]^T.Beta[0]^T`).

## 4. The algorithm

### 4.0 Deviation from scalar minres's exact bookkeeping (deliberate, stated up front)

Scalar `minres` keeps **unnormalized** Lanczos vectors `r1`/`r2` alongside the normalized `v`, and folds
the normalization scalar `beta`/`oldb` into the three-term recurrence as explicit division factors
(`y -= (beta/oldb).r1`). `bminres` instead uses the **standard textbook block-Lanczos form** (fully
normalized block vectors throughout, no separate unnormalized copy) -- e.g. Saad, *Iterative Methods for
Sparse Linear Systems*, 2nd ed., block-Lanczos process (SS6.12 context); Golub & Van Loan, *Matrix
Computations*, block Lanczos. This is mathematically equivalent (both are the same Lanczos recurrence;
scalar `minres`'s r1/r2 trick is just one specific bookkeeping choice) and is simpler to generalize to
block form because the normalization is folded directly into a well-conditioned `s x n` block rather than
needing extra `s x s` scaling matrices threaded through the three-term recurrence. State this explicitly in
the code's contract comment on `bminres` (one line -- detailed rationale goes in DEVLOG, not code, per
`CLAUDE.md`).

### 4.1 Block Lanczos recurrence (produces `Alfa[k]`, `Beta[k]`, `V[k+1]` from `V[k]`, `V[k-1]`, `Beta[k-1]`)

Block-Lanczos three-term recurrence, `V[k]` always `s x n` with (approximately) orthonormal rows
(`V[k].V[k]^T ~= I_s`, or M^-1-orthonormal rows under preconditioning, SS5):

```
Wk := A . V[k]                                    // ApplyBlock, s x n
if k >= 1:  Wk -= BlockCTV(Beta[k-1], V[k-1])     // Wk -= Beta[k-1]^T . V[k-1]  (s x n)
Alfa[k] := BlockGram(V[k], Wk)                    // s x s, symmetric (A symmetric, V[k] orthonormal)
Wk -= BlockCTV(Alfa[k], V[k])                     // orthogonalize against the current block
Beta[k], V[k+1] := BlockNormalize(Wk)             // ridge-regularized; see below
```

`BlockNormalize(ref fProxyMxN W, ref fProxyMxN Z, bool useZ, ref fProxyMxN Beta, int s)` -- new private
helper, structurally mirrors `BlockSolveSPD`'s ridge-retry loop but **keeps the factor** instead of only
using it to solve:
1. If `!useZ` (unpreconditioned): `Z` aliases `W` (same buffer/handle). If `useZ` (preconditioned, SS5):
   caller has already set `Z := M^-1.W` via `BlockApplyPre` before calling.
2. `G := BlockGram(W, Z, s)` (`s x s`; symmetric since `M` is SPD or `M=I`).
3. Ridge-Cholesky-factor `G` into a `work` scratch exactly like `BlockSolveSPD`'s existing loop
   (`diagMax` scale, escalating ridge `16^attempt . Consts.fProxyEpsilon . diagMax`, up to 6 attempts,
   `CHO.decompInPlace`) -- copy/adapt that loop rather than re-deriving it; it is the SAME defensive
   pattern for the SAME reason (a rank-deficient/near-dependent block).
4. `Beta := CopyMat(work, s)` (the `s x s` lower-triangular Cholesky factor -- this is the "keep the factor"
   difference from `BlockSolveSPD`, which only returns the solved RHS).
5. `CHO.decompSolve(ref work, ref Z)` -- solves `work.Z = Z` in place, i.e. `Z := Beta^-1.Z`; `Z` now holds
   the orthonormalized new block-Lanczos vector. (When `!useZ`, this also mutates `W`'s buffer since `Z`
   aliases it -- `W`'s raw contents are not needed again after this call, matching how block-CG's own `Q`
   scratch is freely overwritten.)
6. Returns `false` only if step 3's ridge retry is exhausted (mirrors `BlockSolveSPD`) -- caller treats this
   as `Breakdown`.

At `k=0` (setup, SS4.4) this is called on `R0 = B - A.X0` directly (no `Alfa`/prior-`Beta` subtraction --
there is no `V[-1]`), producing `Beta[0]`, `V[1]`.

### 4.2 The block Givens/Householder completion (`Omega`, the "banded-Givens" high-effort part)

Scalar minres carries one 2x2 rotation `(cs, sn)` per iteration, applied as the SYMMETRIC orthogonal
matrix `R = [[cs,sn],[sn,-cs]]` (a reflection: `R = R^T = R^-1`) to a 2-vector each step. The direct block
generalization needs a **full 2s x 2s orthogonal matrix** `Omega` each iteration (not just its action on one
vector), because `Omega^T` must be applied consistently to THREE different `2s`-tall quantities per
iteration: the `(Dbar, Alfa | 0, Beta)` block-2x2 (giving `Delta`/`Gbar`/`Epsln`/new `Dbar`), the
`(Phibar, 0)` stack (giving `Phi`/new `Phibar`), and it determines `Gamma` itself.

Given `Gbar` (`s x s`, computed in SS4.3) and `Beta[k]` (`s x s`, from SS4.1), form the `2s x s` stack:

```
Y := [ Gbar ]      (rows 0..s-1 = Gbar, rows s..2s-1 = Beta[k], 2s x s)
    [ Beta ]
```

1. **Thin QR of `Y`:** `Gamma, Qy := QR.decomp(Y)` -- `Qy` is `2s x s` with orthonormal columns, `Gamma` is
   `s x s` (generally full/dense, need NOT be triangular at the block level -- only `Y`'s own internal `2s`
   scalar rows get triangularized by the underlying scalar Householder QR, which is irrelevant here; only
   the BLOCK-level zero pattern `Qy^T.Y = Gamma`, `Qperp^T.Y = 0` matters).
2. **Orthogonal completion (`Qperp`, `2s x s`, so `[Qy | Qperp]` is `2s x 2s` orthogonal):**
   - Seed `Z0 := [0_{sxs} ; I_s]` (`2s x s`: top `s` rows zero, bottom `s` rows identity).
   - Project out `Qy`'s component: `T := Qy^T.Z0` (`s x s`, via `Blas.dot(Qy, Z0, ref T, true, false)`);
     `Z1 := Z0 - Qy.T` (`2s x s`, via `Blas.dot(Qy, T, ref QyT, false, false)` then subtract).
   - `Qperp, Rz := QR.decomp(Z1)`. Check rank: if any `|Rz[i,i]| <= Consts.fProxyZeroThreshold`, `Z0`'s
     column space overlapped `Qy`'s too much -- retry once with the OTHER seed `Z0 := [I_s ; 0_{sxs}]`
     (top identity instead of bottom). If **both** seeds give a rank-deficient `Rz`, this is a genuine
     breakdown (extremely degenerate `Y`) -- return `Breakdown`.
   - `Omega := [Qy | Qperp]` (`2s x 2s`: columns `0..s-1` = `Qy`, columns `s..2s-1` = `Qperp` -- a plain
     column-block copy into a `2s x 2s` `Temp` buffer).
3. **Gauge freedom (state this explicitly so the coder doesn't chase a phantom bug):** the SPECIFIC choice
   of `Qperp` among all valid orthogonal completions of `Qy` is not unique, and does not need to be. Any
   valid orthogonal `Omega` gives a mathematically correct `X` -- the internal `Dbar`/`Epsln`/`Phibar`
   values it produces are an implementation-internal bookkeeping choice, never observed by the caller.
   Correctness depends only on: (a) `Omega` being genuinely orthogonal, (b) the SAME `Omega` (via `Omega^T`)
   being applied consistently to all three quantities that need it each iteration (SS4.3), and (c) the
   mandatory verify-at-exit (SS4.5) never claiming `Converged` on a stale/wrong residual.

### 4.3 Coupled two-term update (per iteration `k = 0, 1, 2, ...`)

State carried in from the previous iteration (or SS4.4's setup at `k=0`): `Dbar` (`s x s`), `Epsln` (`s x s`),
`Phibar` (`s x s`, SS3's orientation), `OmegaOld` (`2s x 2s`), `V[k]` (current), `V[k-1]` (previous, unused at
`k=0`), `Beta[k]` (from setup or the previous iteration's `BlockNormalize`), `W1`, `W2` (search-direction
history, zero-initialized at setup).

```
# ---- Lanczos step (produces Alfa[k], Beta[k+1], V[k+1]) -- SS4.1 ----
Wk := A.ApplyBlock(V[k])
if k >= 1: Wk -= BlockCTV(Beta[k], V[k-1])
Alfa := BlockGram(V[k], Wk)
Wk -= BlockCTV(Alfa, V[k])
ok, BetaNext, Vnext := BlockNormalize(Wk, Z, useZ=!M.IsIdentity, ...)
if !ok: status = Breakdown; iterations = k; goto cleanup

# ---- apply the OLD 2s x 2s Omega to the stacked block-2x2 (Dbar,0 ; Alfa,Beta[k]) ----
M2 := assemble 2s x 2s: top-left=Dbar, top-right=0, bottom-left=Alfa, bottom-right=Beta[k]
Result := Blas.dot(OmegaOld, M2, ref Result, transposeA:true, transposeB:false)   // OmegaOld^T . M2
Delta     := Result[0:s, 0:s]
EpslnNext := Result[0:s, s:2s]
Gbar      := Result[s:2s, 0:s]
DbarNext  := Result[s:2s, s:2s]
OldEps := Epsln                    // save BEFORE overwriting, mirrors scalar's "oldeps = epsln" at loop top
Epsln  := EpslnNext
Dbar   := DbarNext

# ---- new Omega from (Gbar, BetaNext), finalizing Gamma -- SS4.2 ----
OmegaNew, Gamma := BuildOmega(Gbar, BetaNext)     // (or Breakdown, per SS4.2 step 2's both-seeds-fail case)

# ---- RHS update ----
PhibarStack := assemble 2s x s: top=Phibar, bottom=0
Res2 := Blas.dot(OmegaNew, PhibarStack, ref Res2, transposeA:true, transposeB:false)  // OmegaNew^T . PhibarStack
Phi        := Res2[0:s, :]
PhibarNext := Res2[s:2s, :]
Phibar := PhibarNext

# ---- search-direction update: solve Gamma . Wnew = V[k] - OldEps^T.W1 - Delta^T.W2  (chosen convention) ----
RHS := V[k] - BlockCTV(OldEps, W1) - BlockCTV(Delta, W2)     // s x n
GammaCopy := copy of Gamma (LU.solveInPlace destroys its input)
pivS := new Pivot(s, Allocator.Temp)
info := LU.solveInPlace(ref GammaCopy, ref pivS, ref RHS)    // RHS becomes Wnew in place
pivS.Dispose()
if info.status != DirectSolveStatus.Success: status = Breakdown; iterations = k; goto cleanup
Wnew := RHS

# ---- X update ----
T := BlockCTV(Phi, Wnew)           // s x n  (Phi^T . Wnew)
X += T                             // BlockAdd(ref X, T, +1)

# ---- roll the 3-buffer rotations (LOCAL ref-parameter swaps -- safe, see SS4.6) ----
{ swap Vprev <-> Vcur; swap Vcur <-> Wk-holding-Vnext }   // Vprev:=old Vcur, Vcur:=Vnext
{ swap W1 <-> W2; swap W2 <-> W-slot; W-slot := Wnew }    // (W1,W2,Wnew) rotate like scalar's (w1,w2,w)
Beta[k+1] := BetaNext
OmegaOld := OmegaNew

# ---- cheap per-column convergence probe, then MANDATORY verify-at-exit (see SS4.5) ----
phibarNormSq[j] := sum_i PhibarNext[i,j]^2   for each of the s ORIGINAL RHS columns j
if phibarNormSq[j] <= thr[j] for ALL j: verify (SS4.5); if verified, status=Converged, iterations=k+1, goto cleanup
```

**Pairing convention, chosen and self-consistent (flag for the coder, verify via SS7's s=1 test):**
`OldEps` pairs with `W1`, `Delta` pairs with `W2` -- this mirrors scalar minres's own pairing exactly
(`w = (v - oldeps.w1 - delta.w2)/gamma`; note `oldeps` multiplies `w1`, not `w2` -- do not swap these).
`Gamma`'s solve direction (`Gamma.Wnew = RHS`, no transpose) and the `BlockCTV`/`C^T.X` convention used
throughout for `Alfa`/`Beta`/`OldEps`/`Delta`/`Phi` are internally-consistent choices with one degree of
gauge freedom each (SS4.2 point 3) -- they are not independently "the" unique correct formulas the way the
Lanczos recurrence (SS4.1) is; **the definitive check that these choices compose correctly is SS7's `s=1`
reduction test**, not a hand-derivation the coder needs to re-verify from a paper.

### 4.4 Setup (k=0, before the loop)

```
thr[j] := tol^2 . sum_c B[j,c]^2            // per original-RHS-column threshold, computed from B, original order
if thr[j] == 0 for ALL j:  X := B; status = Converged; iterations = 0; done   // mirrors scalar bb==0 shortcut

R0 := B - A.ApplyBlock(X0)                                  // s x n, reuse Wk as scratch
if CountConverged(R0, thr, s, n, ...) == s:  status = Converged; iterations = 0; done

ok, Beta[0], V[1] := BlockNormalize(R0, Z, useZ=!M.IsIdentity, ...)
if !ok: status = Breakdown; iterations = 0; done

Phibar := Transpose(Beta[0])     // s x s -- see the orientation note in SS3
Dbar := 0 (s x s);  Epsln := 0 (s x s)
OmegaOld := block-diag(-I_s, I_s)   // 2s x 2s, the k=0 old-rotation seed (matches scalar cs=-1, sn=0)
W1 := 0 (s x n);  W2 := 0 (s x n)
Vprev := unused at k=0 (V[0] does not exist; the k>=1 guard in SS4.3 skips referencing it)
enter the loop at k=0 with V[k]=V[1]
```

Why Phibar is set to Beta transposed, not Beta itself: R0's row-block factorization R0 = Beta[0] times
V[1] (rows of R0 = original RHS, Beta[0] relates them to V[1]'s rows) is the row-major/LQ-style analogue
of the column-major relation the projected least-squares problem actually needs: R0 transposed equals
V[1] transposed times Beta[0] transposed, i.e. the s x s block that seeds the RHS of the reduced
(block-tridiagonal) system is Beta[0] transposed, not Beta[0]. At s=1 this distinction vanishes (a 1x1
matrix equals its own transpose) -- consistent with, and checked by, the s=1 reduction test.

### 4.5 Termination and the MANDATORY verify-at-exit

Scalar minres only re-verifies the residual with a fresh ApplyBlock/Apply under a REAL preconditioner
(phibar is only an M-inverse-weighted proxy there); its identity path trusts phibar directly with no
verify. bminres deviates from that on purpose: because the block bookkeeping (SS4.2-SS4.3) carries more
internal state and more places a sign/pairing convention could be off than the scalar loop does, bminres
always re-verifies before reporting Converged, identity path included -- a fresh R := B - A.ApplyBlock(X)
(reuse Wk or whatever s x n scratch is idle) followed by the existing CountConverged(R, thr, s, n, out
maxRnorm). This makes correctness self-checking: a subtle bookkeeping bug in SS4.2/SS4.3 can only ever
manifest as "doesn't actually converge" (an honest MaxIterations / slow convergence), never as a false
Converged claim with a wrong X. State this deviation and its rationale explicitly in the DEVLOG entry
(SS11 step 8), not as a code comment beyond one line noting verify-at-exit is unconditional here.

Exit paths, every one of which funnels through the fresh verify above before finalizing status/converged
(mirrors bcgrq's SS3.4 cleanup funnel):
- Converged: the cheap per-column phibarNormSq probe (SS4.3, end of loop body) says all s columns are
  within tolerance AND the fresh verify confirms CountConverged(...) == s. If the cheap probe passes but
  the fresh verify does NOT confirm all s, do not return -- fall through and keep iterating (mirrors
  scalar minres's preconditioned-path fallthrough on a failed verify).
- Breakdown: BlockNormalize ridge-exhausted (SS4.1), BuildOmega's both-seed rank check fails (SS4.2), or
  LU.solveInPlace on Gamma returns non-Success (SS4.3).
- MaxIterations: loop reaches maxIter without the Converged condition firing. Still runs the fresh verify
  (for an honest maxRnorm/converged count in the returned BlockSolveInfo, mirroring bcgrq's cleanup and
  cg's/minres's own MaxIterations-path verify).

BlockSolveInfo result: rhs = s, converged = (from the final CountConverged), iterations = (set at the
exit point), maxRnorm = (from the final CountConverged), minActive = s, status = (set at the exit point).

### 4.6 Buffer rotation is safe here (no RestoreBufferIdentity hazard)

The bcgrq spec explicitly avoids ping-ponging block buffer HANDLES because RestoreBufferIdentity
(LOBPCG.fProxy.cs:963) exists to fix exactly that hazard for a cache STRUCT FIELD reseated inside an IJob
(the caller copy of the struct does not see the reseat after the job by-value copy returns). The bminres
Vprev/Vcur/Wk and W/W1/W2 buffers are plain ref fProxyMxN local parameters (not struct fields) -- exactly
like scalar minres own r1/r2/y/w/w1/w2, which already swap freely inside the exact same file
(Krylov.fProxy.cs:633,671). Swapping local ref-bound handles is safe because only X (mutated element-wise
in place via BlockAdd, never reassigned to a different buffer) is ever observed by the caller after the
call returns. Use the same swap-idiom scalar minres uses; do not add a RestoreBufferIdentity-style
buffer-identity-restoring step -- it is not needed and would be dead code.

## 5. Preconditioning

Mirrors scalar minres preconditioned path structurally: the block-Lanczos recurrence runs in the
M-inverse-weighted inner product. The only two places M enters:
- BlockNormalize's useZ branch (SS4.1): Z := M-inverse applied to W, via BlockApplyPre(in M, in W, ref Z,
  s, n, ref rowIn, ref rowOut) (the existing block-CG helper, unmodified) computed by the CALLER before
  invoking BlockNormalize; Beta comes from the Gram of W and Z (M-weighted), and the output block-Lanczos
  vector is built from Z (the preconditioned image), not W -- exactly mirroring scalar's v = z/beta using
  z, not the raw residual.
- Alfa := BlockGram(V[k], Wk) stays a plain EUCLIDEAN Gram in both the identity and preconditioned paths
  -- no M involved there, matching scalar (alfa = dot(v, y), no z).

Every other step (SS4.2's Omega construction, SS4.3's coupled two-term update) is IDENTICAL under
preconditioning -- M only ever touches the normalization step. Gate BlockApplyPre's call and Z's
size/aliasing checks behind if (!M.IsIdentity), exactly like every other merged solver in this codebase;
Z may be default on the identity path (never dereferenced).

Per the task brief: preconditioned bminres must be correct (one test, SS10 PreconditionedMatchesScalar,
and the mandatory IdentityFoldBitIdentical test) but is not a benchmarking priority -- do not add a
bminres row to any preconditioner benchmark as part of this task.

## 6. New private helpers to add to Krylov.Block.fProxy.cs

- `static bool BlockNormalize(ref fProxyMxN W, ref fProxyMxN Z, bool useZ, ref fProxyMxN Beta, ref fProxyMxN G, ref fProxyMxN work, int s)`
  -- SS4.1. G/work/Beta are s x s scratch (Temp or caller-provided, coder choice -- match bcg style of
  allocating small s x s Temp buffers per call, since these are cheap and this mirrors BlockSolveSPD
  existing pattern of local Temp s x s scratch).
- `static void CopyRowsAt(in fProxyMxN src, ref fProxyMxN dst, int rowOffset, int rows, int cols)` --
  copies src first `rows` rows into dst starting at row `rowOffset`. Used to assemble the 2s x s / 2s x 2s
  stacked buffers (Y, Z0, M2, PhibarStack, Omega column-block assembly) from separate s x s / s x n
  sources.
- `static void CopyBlockAt(in fProxyMxN src, int srcRowOff, int srcColOff, ref fProxyMxN dst, int rows, int cols)`
  -- copies the rows x cols submatrix of src starting at (srcRowOff, srcColOff) into dst (from row/col 0).
  Used to extract Delta/EpslnNext/Gbar/DbarNext/Phi/PhibarNext from the 2s x 2s / 2s x s Result/Res2
  buffers, and to extract Qy/Qperp sub-blocks when assembling Omega.
- `static bool BuildOmega(in fProxyMxN Gbar, in fProxyMxN Beta, ref fProxyMxN Omega, ref fProxyMxN Gamma, int s)`
  -- SS4.2, encapsulates the two-QR-call completion recipe (including the both-seeds-fail Breakdown
  check). Owns its own Y/Z0/Z1/Qy/Qperp/Rz/T/QyT Temp scratch internally (all 2s x s or s x s, cheap, do
  not expose them to the caller -- mirrors how BlockSolveSPD owns its ridge-loop scratch internally).

Comment style: contract only (shapes, what it destroys, what it requires), per CLAUDE.md. The Omega
gauge-freedom note (SS4.2 point 3), the Phibar transpose derivation (SS3/SS4.4), and the pairing-
convention rationale (SS4.3) are exactly the kind of "why" content that belongs in OP/DEVLOG.md under a
"Krylov.bminres" heading, not in code comments -- a one-line contract comment (what it solves, that it is
symmetric-indefinite not SPD, that it deflates via ridge not column-dropping) is enough on bminres itself.

## 7. The s=1 reduction test -- the primary correctness oracle (read before implementing SS4.3)

At s=1, every block quantity in SS4.1-SS4.3 is a 1x1 "matrix" and every block operation degenerates to
its scalar counterpart: BlockNormalize reduces to beta=norm(r), v=r/beta; BlockGram reduces to dot;
BuildOmega's Qy is exactly [gbar; beta]/gamma = [cs; sn]; the s=1 Omega is a 2x2 orthogonal matrix (some
valid rotation/reflection, not necessarily bit-identical to scalar's specific [[cs,sn],[sn,-cs]]
construction, per SS4.2 point 3's gauge freedom); and the whole recurrence should trace the SAME sequence
of X iterates as scalar minres on the same single-column system, to floating-point round-off (not
necessarily bit-identical, since the internal Omega/Gamma/Phibar bookkeeping differs by construction even
though the X trajectory is mathematically forced to agree). Write MatchesScalarAtS1 (SS10) as an early,
cheap, high-signal test -- implement it and get it passing BEFORE chasing any s>1 test failures. If the
pairing/orientation choices in SS4.3 or SS4.4 are wrong, this is where it shows up first and most legibly.

## 8. Argument validation

Mirror bcg's block-CG core validation pattern: A.Rows == A.Cols (ArgumentException); s = B.M_Rows, n =
A.Rows, B.N_Cols == n; X/Vprev/Vcur/Wk/W/W1/W2 all s x n; Z is s x n only required if !M.IsIdentity; s <=
n (ArgumentException, SS3); maxIter >= 1. Aliasing guard: all s x n scratch buffers
(Vprev,Vcur,Wk,W,W1,W2,Z,X,B) pairwise distinct, via RequireDistinctBuffers (OP/Krylov.Guards.cs:14), same
pattern as scalar minres's own guard (Krylov.fProxy.cs:550-560) -- Z joins the checked set only under
!M.IsIdentity.

## 9. Public API -- overload ladder (mirrors bcg's / bcgrq's ladder structure)

```csharp
// 1. Generic core.
public static BlockSolveInfo bminres<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                ref fProxyMxN Vprev, ref fProxyMxN Vcur, ref fProxyMxN Wk,
                                ref fProxyMxN W, ref fProxyMxN W1, ref fProxyMxN W2, ref fProxyMxN Z,
                                int maxIter, fProxy tol)
    where TOp : struct, IfProxyLinearOperator
    where TPre : struct, IfProxyPreconditioner

// 2. Unpreconditioned forwarder (default(fProxyIdentityPreconditioner), default Z -- never dereferenced
//    under the IsIdentity fold, exactly like cg<TOp>/bcgrq<TOp>).
public static BlockSolveInfo bminres<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                ref fProxyMxN Vprev, ref fProxyMxN Vcur, ref fProxyMxN Wk,
                                ref fProxyMxN W, ref fProxyMxN W1, ref fProxyMxN W2,
                                int maxIter, fProxy tol)
    where TOp : struct, IfProxyLinearOperator

// 3. Dense (symmetric, possibly indefinite), arena-allocating.
public static BlockSolveInfo bminres(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)

// 4. Dense, default maxIter (A.M_Rows) / tol (Consts.fProxySqrtEps).
public static BlockSolveInfo bminres(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)

// 5. Dense, preconditioned, arena-allocating (also allocates Z).
public static BlockSolveInfo bminres<TPre>(in fProxyMxN A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                int maxIter, fProxy tol)
    where TPre : struct, IfProxyPreconditioner

// 6. BSR (symmetric), arena-allocating.
public static BlockSolveInfo bminres(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)

// 7. BSR, default maxIter/tol.
public static BlockSolveInfo bminres(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)

// 8. BSR, preconditioned, arena-allocating.
public static BlockSolveInfo bminres<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                int maxIter, fProxy tol)
    where TPre : struct, IfProxyPreconditioner
```

Arena-allocating rungs (3, 4, 5, 6, 7, 8) allocate Vprev,Vcur,Wk,W,W1,W2[,Z] via B.fProxyTempMat(s, n,
true), same convention as bcg's dense/BSR convenience overloads.

## 10. Tests -- new file Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BlockMinresTests.fProxy.cs

Mirror BlockCGTests.fProxy.cs's structure exactly: one [BurstCompile(CompileSynchronously = true)] IJob
struct, TestType enum switch, every scenario built and asserted inside Execute(), [Test] methods doing
`new ...Job { Type = ... }.Run()`. Assert.IsTrue(bool)/Assert.AreEqual only -- never the string-message
overload (BC1071, silent Mono fallback -- see CLAUDE.md / memory notes on this exact gotcha).

New helper (replaces BlockCGTests's BuildDenseSPD): BuildDenseSymIndefinite(ref Arena, int dim, uint
seed) -- build a random symmetric matrix with a genuinely INDEFINITE spectrum (e.g. M = RandomMat(dim,
dim), A = (M + M-transpose)/2, then shift the diagonal so eigenvalues span both signs -- do NOT add a
positive diagonal boost the way BuildDenseSPD does, that would make it SPD and defeat the point of testing
MINRES on an indefinite system). Reuse Row/DenseToBSR1x1/Tol() adapted from BlockCGTests.

Required test cases:

1. MatchesScalarAtS1 (SS7) -- s=1, symmetric indefinite A, bminres vs scalar Krylov.minres on the same
   single-column system: matching X to tolerance, matching Solved/converged. Get this passing FIRST.
2. MatchesScalarMinresPerColumn -- s>1 (e.g. s=4), symmetric indefinite dense A; bminres's block solve
   matches s independent scalar Krylov.minres solves per column to tolerance, and info.Solved /
   info.converged == s.
3. KnownSolutionRecovered -- known Xk (s x n), B = A.ApplyBlock(Xk), recover Xk to tolerance, on a
   symmetric INDEFINITE A.
4. BlockAdvantageIterations -- mirrors BlockCGTests.BlockAdvantageIterations: bminres's block iteration
   count is <= the worst single-column scalar minres iteration count on the same well-separated (but
   still indefinite) system.
5. RankDeficientBlockNoNaN -- two RHS columns forced exactly identical (mirrors
   BlockCGTests.RankDeficientBlockDeflates); assert finite/no-NaN throughout X, info.Solved, matching
   columns equal, and (per SS2's scope decision) info.minActive == info.rhs (v1 does NOT drop columns --
   this is the concrete difference from a future bminresrq, note it in the test comment/DEVLOG, not as a
   surprising assertion).
6. PreconditionedMatchesScalar -- BSR + block-Jacobi preconditioner (fProxyBlockJacobi), symmetric
   indefinite BSR system, matches per-column scalar preconditioned Krylov.minres(in A, in M, ...).
7. IdentityFoldBitIdentical -- bminres<TOp, fProxyIdentityPreconditioner> (explicit identity) produces
   BIT-IDENTICAL X/iterations/status to the unpreconditioned bminres<TOp> overload on the same fixed-seed
   system (exact double-equality, no tolerance) -- mirrors bcg's/bcgrq's own IdentityFoldBitIdentical-
   style test.
8. JobSafeThroughRun -- satisfied by construction (every test above runs via IJob.Run()); confirm in
   review that X is only ever mutated element-wise (never reassigned to a different buffer) -- SS4.6's
   invariant -- so there is no RestoreBufferIdentity-style hazard to test for; no extra test needed beyond
   this review note.

All tests use Consts.fProxySqrtEps-scale tolerances and the Tol()/Row/DenseToBSR1x1 private-helper style
copied from BlockCGTests.fProxy.cs (adapt, do not import cross-file).

## 11. Implementation checklist (ordered)

1. Add BlockNormalize, CopyRowsAt, CopyBlockAt, BuildOmega private helpers to Krylov.Block.fProxy.cs
   (SS6).
2. Implement the generic core bminres<TOp, TPre> per SS4 exactly (setup SS4.4, per-iteration SS4.3, using
   SS4.1's Lanczos step and SS4.2's BuildOmega, cleanup/verify-at-exit SS4.5).
3. Add the 7 forwarding/convenience overloads per SS9.
4. Regenerate (Tools/regen.ps1) and confirm float+double both compile clean.
5. Write BlockMinresTests.fProxy.cs (SS10). Implement and pass test 1 (MatchesScalarAtS1) BEFORE moving on
   to the s>1 tests (SS7) -- it is the cheapest, highest-signal debugging target for the pairing/
   orientation choices in SS4.3/SS4.4.
6. Run the full suite headlessly; confirm the exact line `Result=Passed total=N passed=N failed=0` (never
   pipe through `| tail`).
7. Add a "Krylov.bminres" DEVLOG.md entry (dated, newest-first, per CLAUDE.md's format): the s=1-reduction
   verification outcome, the deliberate always-verify-at-exit deviation from scalar minres (SS4.5) and
   why, the Omega gauge-freedom note (SS4.2), and -- explicitly -- the deferred true column-dropping
   deflation as a future bminresrq follow-up (mirroring bcg to bcgrq). Do not put any of this in code
   comments.

## 12. Acceptance criteria

- Krylov.bminres exists in OP/Krylov.Block.fProxy.cs with the 8-overload ladder of SS9, generated cleanly
  for both float and double.
- Existing block-CG (cg/bcg) code and its reused helpers (BlockGram, BlockCTV, BlockAdd, CopyBlock,
  CopyMat, BlockApplyPre, CountConverged) are UNMODIFIED (same signatures, same bodies) -- no signature
  changes to any existing helper.
- All 8 tests in BlockMinresTests.fProxy.cs (SS10) exist and pass, including MatchesScalarAtS1 (SS7) and
  IdentityFoldBitIdentical (exact, non-tolerance equality).
- bminres never claims IterativeSolveStatus.Converged without a fresh CountConverged(...) == s
  confirmation at the reported X (SS4.5) -- this is checkable indirectly via test 2/3's info.Solved +
  per-column residual assertions, and directly by code review of the exit paths.
- RankDeficientBlockNoNaN passes with info.minActive == info.rhs (v1's fixed-width scope, SS2).
- Full project test suite green: the literal line `Result=Passed total=N passed=N failed=0` from the
  headless test run, N including the 8+ new bminres tests, failed=0.
- No edits to README.md. No edits to anything under Assets/LinearAlgebra/Source/ (generated output --
  regenerate instead). No edits to Pivot/Pivot.cs / Pivot/Pivot.Operations.cs (priority-backlog item 2,
  out of scope here -- bminres only uses Pivot's existing public surface via LU.solveInPlace).
- A "Krylov.bminres" DEVLOG.md entry exists per SS11 step 7.

## 13. Out of scope (do not do these in this task)

- True rank-revealing column-dropping deflation of the block-Lanczos recurrence (variable width sa < s
  mid-solve) -- deferred to a future bminresrq-style follow-up, mirroring bcg to bcgrq. bminres v1's
  deflation is ridge-only, fixed width s throughout (SS2).
- Any preconditioner benchmark row for bminres (SS5) -- correctness only, not benchmarked, per the task
  brief.
- Block-BiCGStab / block-GMRES (items 3-4 of docs/dev/spec-block-krylov.md SS5) -- not touched.
- Resolving the Pivot "Arena dependency?" TODO (priority-backlog item 2) -- bminres uses Pivot exactly as
  it exists today (one `new Pivot(s, Allocator.Temp)` per iteration for the Gamma LU solve, mirroring
  bcgrq's own per-iteration Pivot allocation for its LQRP calls).
- Renaming block-CG from cg to bcg -- unrelated, tracked separately; this task does not depend on it
  either way (SS0).
- Any change to BlockSolveInfo, bcg's block-CG core, or bcgrq (whether or not bcgrq has been implemented
  yet by the time this task runs).
- SVD, least squares, optimizers, sparse-matrix work, View/Slice -- unrelated to this task.
