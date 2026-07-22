# Mini-spec: `Krylov.bbiCGStab` -- block (multi-RHS) BiCGSTAB for nonsymmetric systems

## 0. Naming note (read this first)

`bbiCGStab` (block-prefix `b` + scalar name `biCGStab`) is the working name per the project's
lowercase-`b`-prefix block-method convention (`cg` -> `bcg`, and the planned `cgrq` -> `bcgrq`). The
doubled `b` (`b` + `biCGStab`) is visually awkward -- `bbiCGStab` reads as "bbi-CG-Stab" on first
glance, not "b-BiCGStab". **Suggested alternative: `bBiCGStab`** (capitalize the remainder's leading
letter after the block prefix, i.e. `b` + `BiCGStab`), which reads more clearly as a block prefix over a
proper-noun method name and is not a bigger departure from `bcg`'s own casing than `bbiCGStab` is. **This
spec targets the working name `bbiCGStab` as instructed** -- if the coder or a later pass prefers
`bBiCGStab`, that is a pure rename with no semantic change; do not treat this as blocking.

## 1. Task

Add `Krylov.bbiCGStab` -- a TRUE block-Krylov BiCGSTAB solver for nonsymmetric square systems `A X = B`
with `s` simultaneous right-hand sides -- to `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.fProxy.cs`,
alongside (not replacing) the existing block-CG (`Krylov.bcg`). One shared block Krylov subspace built
from all `s` RHS at once (`s x s` matrix coefficients, one `ApplyBlock` per matvec), not `s` independent
scalar `biCGStab` solves.

## 2. Why now / context already read

- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.fProxy.cs:811-963` -- scalar
  `biCGStab<TOp,TPre>`. This spec's block recurrence is a direct, line-by-line structural mirror of this
  loop (same variable roles, same breakdown-check placement, same `rho=alpha=omega=1,p=v=0` zero-init
  trick that makes iteration 0 fall out of the general loop with no special-casing). Read it before
  reading Section 5 below.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.fProxy.cs` -- **current state (verified by
  reading the file directly, not assumed): block-CG has ALREADY been renamed from `cg` to `bcg`** (an
  8-rung overload ladder: generic core `bcg<TOp,TPre>`, unpreconditioned `bcg<TOp>`, dense
  arena/default/preconditioned, BSR arena/default/preconditioned). `bbiCGStab` goes in this same file,
  mirrors this exact 8-rung ladder shape, and reuses these existing private helpers UNMODIFIED:
  `BlockGram`, `BlockCTV`, `BlockAdd`, `BlockZplusT`, `CopyBlock`, `CopyMat`, `BlockApplyPre`,
  `CountConverged`. `BlockSolveSPD` (Cholesky-based) is **not** reused -- see Section 7, a new QRCP-based
  general solve is needed instead (SPD is not assumed here).
- `docs/dev/spec-bcgrq.md` and `docs/dev/spec-block-krylov.md` -- block deflation/orientation
  conventions. Confirmed by re-deriving from `BlockGram`/`BlockCTV`'s actual bodies (not just the docs):
  with block vectors stored `s` ROWS x `n` COLS (row `j` = classical column `j` of an `n x s` block),
  `BlockGram(V, W, G, s)` computes exactly the classical Gram `G = V_classical^T W_classical`, and
  `BlockCTV(C, V, dst)` computes exactly the classical `dst = V_classical . C`. **This means the `s x s`
  coefficient matrices this spec solves (`rho`, `M`, the beta-prep system) need NO transpose dance at the
  small-matrix boundary** -- `BlockSolveSPD` in `bcg` already relies on this same fact (no transpose in
  its body) and this spec follows the identical pattern, just with a nonsymmetric (QRCP) solve instead of
  Cholesky.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QRCP.fProxy.cs:1431-1489` --
  `QRCP.solveInPlace(ref A_to_Q, ref B, ref X, ref R, ref Pivot, ref u, relTol)`: general (nonsymmetric)
  multi-RHS solve via column-pivoted QR. **DESTROYS both `A_to_Q` and `B`**; `X` must be a separate
  buffer. Returns `RankInfo { status, rank }` -- per its own doc, only ever `Success` (full rank) or
  `RankDeficient` (rank < n) for this overload, never a hard failure. `relTol < 0` selects the
  library-standard auto default (`max(m,n) * Consts.fProxyZeroThreshold`). This is the general-solve
  workhorse for every `s x s` system in this spec (Section 4 / Section 7's `BlockSolveGeneral`).
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SolveStatus.cs` -- `RankInfo.status ==
  DirectSolveStatus.Success` vs `RankDeficient`; `IterativeSolveStatus.Breakdown` is what this solver
  returns on any of the checks in Section 4.
- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/BlockSolveInfo.cs` -- return type, unchanged.
- `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BlockCGTests.fProxy.cs` -- test-file style to
  mirror exactly (one `[BurstCompile] IJob` with a `TestType` switch, `[Test]` methods each doing `new
  ...Job{Type=...}.Run()`, `BuildDenseSPD`/`Row`/`DenseToBSR1x1`-style private static helpers).
- `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/SchwarzPreconditionerTests.fProxy.cs:203-238`
  -- `arena.fProxyRestrictedSchwarz` (RAS) is the scalar file's dedicated nonsymmetric-native
  preconditioner, paired with `Krylov.biCGStab` (not `cg` -- RAS is not symmetric). Use it for Section
  10's preconditioned test. `arena.fProxyRandomSparse(blockRows, blockCols, BR, density, seed)`
  (`Gallery.Sparse.fProxy.cs:111`) builds a nonsymmetric, diagonally-dominant-by-construction random BSR
  matrix -- reuse for the BSR test.
- Priority-backlog item 2 (`Pivot`'s "Arena dependency?" TODO) is **not** touched by this task --
  `Pivot` is used exactly as it exists today (matching `spec-bcgrq.md`'s own stance on this).

## 3. Data model and shapes

Same convention as `bcg`: a block vector is an `fProxyMxN` with `s` ROWS x `n` COLS, row `j` =
RHS/solution vector `j` (length `n = A.Rows`). No `s <= n` constraint is required (unlike `bcgrq`, which
needs it for `LQRP`'s row-rank factorization) -- every matrix this solver factors is `s x s`, never
`s x n`, so there is no row-rank-of-a-tall-block concern here.

## 4. The recurrence

Reference family (cited for the algorithm's shape and the block-BiCGSTAB idea -- NOT claimed as a
verbatim equation-for-equation reproduction; this spec's exact formulas are re-derived from the scalar
recurrence, see Section 6 for the explicit honesty note on the one genuinely open point): El Guennouni,
Jbilou, Sadok, "A block version of BiCGSTAB for linear systems with multiple right-hand sides", ETNA 16
(2003), 129-142; the general block-Krylov breakdown/deflation framing already used by
`docs/dev/spec-block-krylov.md` and `docs/dev/spec-bcgrq.md` (O'Leary 1980; Dubrulle 2001; Ji and Li
2017).

### 4.1 Shapes at a glance (classical `n x s` orientation; stored transposed as `s x n` per Section 3)

| classical quantity | shape | our storage | role |
|---|---|---|---|
| `X`, `R` (residual), `Rhat0` (shadow, FIXED), `P` (search dir), `V=A.(pre P)`, `T=A.(pre S)` | `n x s` | `s x n` | block vectors, public buffers |
| `Phat = M^-1 P`, `Shat = M^-1 S` | `n x s` | `s x n` | preconditioner images, public buffers, unused under identity |
| `rho` (`= rho_(k-1)`, carried), `rhoNew` (`= Rhat0^T R_k`), `M` (`= Rhat0^T V_k`), `Y`, `alphaMat`, `betaMat` | `s x s` | `s x s` | coefficient matrices, internal `Allocator.Temp` |
| `omega` | scalar | `fProxy` | shared scalar stabilization coefficient (Section 5) |

### 4.2 Setup (before the loop)

```
n = A.Rows, s = B.M_Rows.  Validate (Section 6).
thr[j] = tol*tol * sum_c B[j,c]^2                          // per-column threshold, original order
R := B - A.X                       (A.ApplyBlock(X, T, s) as scratch; R[i,c] = B[i,c]-T[i,c])
lastConverged, lastMaxRnorm := CountConverged(R, thr, s, n) // initial residual
if lastConverged == s: status=Converged, iterations=0, done
Rhat0 := copy(R)                    (CopyBlock -- FIXED shadow residual for the whole solve)
P := 0 (s x n) ; V := 0 (s x n)      (explicit double loop)
rho := Identity_s ; alphaMat := Identity_s ; omega := (fProxy)1
```
(`rho=alpha=omega=1, p=v=0` is scalar `biCGStab`'s own init trick, generalized to `Identity_s`/block-zero
-- verified below to make iteration `k=0` reduce to `P_0 = R_0` with no special-casing, exactly like
scalar.)

### 4.3 Per-iteration body (`k = 0 .. maxIter-1`)

```
S1.  BlockGram(Rhat0, R, rhoNew, s)                          // rhoNew = Rhat0^T R      (rho_k, s x s)

S2.  if BlockFrobDot(rhoNew,rhoNew) == 0 || isnan(...):
        -> Breakdown, iterations=k, report (lastConverged, lastMaxRnorm)  // mirrors scalar's exact
                                                                            // `rhoNew==0` check (no
                                                                            // tolerance fuzz -- exact
                                                                            // zero/NaN, same as scalar)

S3.  rankY = BlockSolveGeneral(rho, rhoNew, ref Y, ...)       // solve  rho * Y = rhoNew
     if rankY.status != Success: -> Breakdown, iterations=k, report (lastConverged, lastMaxRnorm)

S4.  Blas.dot(in Y, in alphaMat, ref betaMat, false, false)   // betaMat = Y * alphaMat   [see Section 6]
     BlockScaleInPlace(ref betaMat, 1/omega)                  // betaMat *= 1/omega

S5.  BlockAdd(ref P, in V, -omega)                            // P := P_old - omega_old * V_old
S6.  BlockCTV(in betaMat, in P, ref Tmp)                      // Tmp = P_classical * betaMat
S7.  BlockZplusT(in R, in Tmp, ref P)                         // P := R_k + Tmp            (P_new)

S8.  if M.IsIdentity: A.ApplyBlock(in P, ref V, s)
     else: BlockApplyPre(in M, in P, ref Phat, s, n, ref rowIn, ref rowOut)
           A.ApplyBlock(in Phat, ref V, s)                    // V := A * (pre P)          MATVEC #1

S9.  BlockGram(Rhat0, V, Mmat, s)                             // Mmat = Rhat0^T V

S10. rankA = BlockSolveGeneral(Mmat, rhoNew, ref alphaMat, ...)  // solve  Mmat * alphaMat = rhoNew
     if rankA.status != Success: -> Breakdown, iterations=k, report (lastConverged, lastMaxRnorm)
                                                                // X/P/V changed this iter but NOT
                                                                // committed via alpha -- last committed
                                                                // X still matches (lastConverged,
                                                                // lastMaxRnorm) from the PREVIOUS
                                                                // iteration's end (or setup, at k=0)

S11. BlockCTV(in alphaMat, in P, ref Tmp); BlockAdd(ref X, in Tmp, (fProxy)1)   // X += P * alphaMat
S12. BlockCTV(in alphaMat, in V, ref Tmp); BlockAdd(ref R, in Tmp, (fProxy)(-1)) // R := R - V*alphaMat
                                                                // R now holds S (half-step residual)

S13. sConv, sMaxRnorm := CountConverged(R, thr, s, n)
     lastConverged, lastMaxRnorm := sConv, sMaxRnorm           // X now matches this R (=S)
     if sConv == s: -> Converged, iterations=k+1, done         // early exit: skip matvec #2 entirely
                                                                // (mirrors scalar's ss<=threshold exit)

S14. if M.IsIdentity: A.ApplyBlock(in R, ref T, s)
     else: BlockApplyPre(in M, in R, ref Shat, s, n, ref rowIn, ref rowOut)
           A.ApplyBlock(in Shat, ref T, s)                     // T := A * (pre S)          MATVEC #2

S15. tS := BlockFrobDot(T, R)                                  // trace(T^T S)
     tT := BlockFrobDot(T, T)                                  // trace(T^T T)
     if !(tT > 0): -> Breakdown, iterations=k, report (lastConverged, lastMaxRnorm)  // NaN-safe;
                                                                // X still matches S (S13's values)
     omega := tS / tT
     if omega == 0 || isnan(omega): -> Breakdown, iterations=k, report (lastConverged, lastMaxRnorm)

S16. BlockAdd(ref X, in R, omega)                               // X += omega * S   (R still holds S)
S17. BlockAdd(ref R, in T, -omega)                               // R := S - omega*T  (new residual)

S18. rConv, rMaxRnorm := CountConverged(R, thr, s, n)
     lastConverged, lastMaxRnorm := rConv, rMaxRnorm
     if rConv == s: -> Converged, iterations=k+1, done

S19. CopyMat(in rhoNew, ref rho, s)                              // roll: rho := rhoNew for next iter
     // alphaMat, omega already hold this iteration's fresh values -- carried automatically.
```
Loop to `k+1`, or if `k+1 == maxIter`: `status = MaxIterations`, `iterations = maxIter`, report
`(lastConverged, lastMaxRnorm)`.

### 4.4 Verifying the `k=0` reduction (do this sanity check before implementing further)

At `k=0`: `rho=Identity_s` so S3 gives `Y=rhoNew` trivially (never rank-deficient); S4 gives
`betaMat = rhoNew * Identity_s / 1 = rhoNew`; S5 gives `P := 0 - omega*0 = 0` (P,V both start zero); S6
gives `Tmp = BlockCTV(betaMat, 0) = 0`; S7 gives `P := R_0 + 0 = R_0`. This exactly matches scalar
`biCGStab`'s `p_0 = r_0` fallout from its own `p=v=0, rho=alpha=omega=1` init -- confirming the
generalization is structurally sound at the boundary, independent of Section 6's open risk (which only
matters from `k=1` onward, once `rho` stops being `Identity_s`).

## 5. Why scalar omega (not block omega)

`omega` is a single **scalar**, shared across all `s` columns, computed via the Frobenius/trace
minimization `omega = trace(T^T S) / trace(T^T T)` -- the block generalization of scalar's local
`omega = <t,s>/<t,t>` residual-minimization, and the choice used by the El Guennouni-Jbilou-Sadok block
BiCGSTAB family cited above. Justification for scalar over a matrix `omega`:

1. **Shared-subspace structure.** A block `omega` (an `s x s` right-multiplying coefficient, requiring
   its own `T^T T . omega = T^T S` solve) would give each column its own stabilization polynomial, which
   defeats the entire premise of a block method -- the shared Krylov subspace argument (why block methods
   converge in <= the worst single-column iteration count) relies on ONE shared scalar polynomial applied
   uniformly to the whole block, exactly the same structural reason `bcgrq`'s beta derivation
   (`docs/dev/spec-bcgrq.md` Section 5) needs `Pa` to be a single shared orthonormal basis, not a
   per-column one.
2. **Extra singularity risk avoided.** `T^T T` (classical) is symmetric PSD but can be singular/
   ill-conditioned exactly when `S` is nearly rank-deficient (the RHS-block near-parallel case) -- the
   scalar Frobenius ratio sidesteps needing to solve/invert that matrix at all.
3. **One fewer `s x s` solve per iteration.** Scalar `omega` is a pure reduction (`BlockFrobDot`, twice);
   a block `omega` would be a THIRD general `s x s` solve per iteration on top of Section 4's two.

## 6. Honesty note: the one open risk in this derivation

Every operation in Section 4 has been checked for **shape consistency** (every multiply/solve typechecks)
and for **correct reduction at `k=0`** (Section 4.4). The one place this spec's derivation is NOT
independently verified against a published equation is **S4's matrix multiplication order**,
`betaMat = Y * alphaMat` (as opposed to `alphaMat * Y`). Scalar BiCGSTAB's
`beta = (rhoNew/rho)*(alpha/omega)` is built from commuting scalars, so the classical formula does not by
itself disambiguate a matrix ordering; a fully rigorous derivation would need the same combined-polynomial
argument van der Vorst (1992) uses to justify scalar BiCGSTAB's rho/alpha/omega recurrence in the first
place, generalized to matrix coefficients -- a proof this spec does not attempt.

**Do not treat this as blocking** -- Section 10's tests are the correctness oracle:

- `MatchesScalarBiCGStabPerColumn` and `KnownSolutionRecovered` (Section 10 #1/#2) use `s >= 3` GENUINELY
  independent nonsymmetric right-hand sides sharing one `A`. If S4's ordering is wrong, the iterates will
  not track a valid BiCGSTAB trajectory and these tests will fail to converge / miss the known solution
  -- they will NOT pass by accident.
- **If `MatchesScalarBiCGStabPerColumn` or `KnownSolutionRecovered` fail to converge (not a tolerance-slop
  failure, a genuine non-convergence/breakdown), the first thing to try is swapping S4 to
  `Blas.dot(in alphaMat, in Y, ref betaMat, false, false)` (i.e. `alphaMat * Y` instead of `Y *
  alphaMat`)** before assuming a different bug. Record whichever order passes in a `## Krylov.bbiCGStab`
  `DEVLOG.md` entry (per `CLAUDE.md`'s comment policy -- this kind of "which order was empirically
  correct" note belongs in DEVLOG, not in a code comment).

## 7. New private helpers (add to `Krylov.Block.fProxy.cs`)

Comments: contract only (shape in, shape out, what it destroys), matching existing helpers' one-line
style. No rationale in code comments -- anything explaining *why* (this section's content) goes in
`OP/DEVLOG.md` under `## Krylov.bbiCGStab`.

```
// General (nonsymmetric) s x s solve: Coef * X = Rhs, via QRCP (rank-revealing). Coef/Rhs preserved
// (copied into coefWork/rhsWork, which QRCP.solveInPlace destroys). X must be distinct from Coef/Rhs.
static RankInfo BlockSolveGeneral(in fProxyMxN Coef, in fProxyMxN Rhs, ref fProxyMxN X,
                                   ref fProxyMxN coefWork, ref fProxyMxN rhsWork,
                                   ref fProxyMxN Rqrcp, ref Pivot Pqrcp, ref fProxyN uQrcp, int s)
{
    CopyMat(in Coef, ref coefWork, s);
    CopyMat(in Rhs, ref rhsWork, s);
    return QRCP.solveInPlace(ref coefWork, ref rhsWork, ref X, ref Rqrcp, ref Pqrcp, ref uQrcp, (fProxy)(-1));
}

// Frobenius inner product sum_i,c U[i,c]*V[i,c] over the whole (contiguous) block -- equals
// trace(U_classical^T V_classical) regardless of row/col storage convention. U, V must be same shape.
static unsafe fProxy BlockFrobDot(in fProxyMxN U, in fProxyMxN V) { /* pointer loop, mirrors BlockAdd's style */ }

// M *= scale over the whole (contiguous) block, in place.
static unsafe void BlockScaleInPlace(ref fProxyMxN M, fProxy scale) { /* pointer loop, mirrors BlockAdd's style */ }
```

`BlockSolveSPD` (Cholesky-based, `bcg`'s helper) is **not reused** -- `bbiCGStab`'s coefficient matrices
(`rho`, `Mmat`) are not SPD (A is nonsymmetric), so every `s x s` solve here goes through
`BlockSolveGeneral` instead. No existing helper's signature or body changes.

## 8. Argument validation (mirror `bcg`'s core, extended to the extra buffers)

`A.Rows == A.Cols`; `B.M_Rows == s`, `B.N_Cols == n (= A.Rows)`; `X`, `R`, `Rhat0`, `P`, `V`, `T` all
`s x n`; `Phat`/`Shat` `s x n` **only required if `!M.IsIdentity`**; `maxIter >= 1`. No `s <= n`
constraint (Section 3). Aliasing guard: `RequireDistinctBuffers` (`Krylov.Guards.cs`) over
`{R, Rhat0, P, V, T, X, B}` (7 pointers) plus `{Phat, Shat}` when `!M.IsIdentity` (9 pointers) -- exact
mirror of scalar `biCGStab`'s own 7/9-pointer guard (`Krylov.fProxy.cs:834-844`).

## 9. Public API -- overload ladder (mirrors `bcg`'s 8-rung ladder; scalar `biCGStab`'s exact parameter names/order)

```
// 1. Generic core.
public static BlockSolveInfo bbiCGStab<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                ref fProxyMxN R, ref fProxyMxN Rhat0, ref fProxyMxN P, ref fProxyMxN V, ref fProxyMxN T,
                                ref fProxyMxN Phat, ref fProxyMxN Shat,
                                int maxIter, fProxy tol)
    where TOp : struct, IfProxyLinearOperator
    where TPre : struct, IfProxyPreconditioner

// 2. Unpreconditioned forwarder (default(fProxyIdentityPreconditioner); Phat/Shat = default, never
//    dereferenced under the IsIdentity fold -- exact mirror of scalar biCGStab<TOp>).
public static BlockSolveInfo bbiCGStab<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X,
                                ref fProxyMxN R, ref fProxyMxN Rhat0, ref fProxyMxN P, ref fProxyMxN V, ref fProxyMxN T,
                                int maxIter, fProxy tol)
    where TOp : struct, IfProxyLinearOperator

// 3. Dense, arena-allocating (R/Rhat0/P/V/T via B.fProxyTempMat(s, n, true)).
public static BlockSolveInfo bbiCGStab(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)

// 4. Dense, default maxIter (A.M_Rows) / tol (Consts.fProxySqrtEps).
public static BlockSolveInfo bbiCGStab(in fProxyMxN A, in fProxyMxN B, ref fProxyMxN X)

// 5. Dense, preconditioned, arena-allocating (also allocates Phat/Shat).
public static BlockSolveInfo bbiCGStab<TPre>(in fProxyMxN A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                int maxIter, fProxy tol)
    where TPre : struct, IfProxyPreconditioner

// 6. BSR, arena-allocating.
public static BlockSolveInfo bbiCGStab(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X, int maxIter, fProxy tol)

// 7. BSR, default maxIter/tol.
public static BlockSolveInfo bbiCGStab(in fProxyBSR A, in fProxyMxN B, ref fProxyMxN X)

// 8. BSR, preconditioned, arena-allocating.
public static BlockSolveInfo bbiCGStab<TPre>(in fProxyBSR A, in TPre M, in fProxyMxN B, ref fProxyMxN X,
                                int maxIter, fProxy tol)
    where TPre : struct, IfProxyPreconditioner
```

This is a generic `<TPre>` ladder like `bcg`'s (NOT the scalar file's per-preconditioner concrete rungs
such as `biCGStab(in fProxyBSR, in fProxyRestrictedSchwarz, ...)`) -- callers supply their preconditioner
type directly to rung 5/8. Dedicated concrete preconditioner rungs are explicitly out of scope
(Section 13).

Internal `Allocator.Temp` scratch declared once before the loop in the core (Section 4 / Section 7):
`rho`, `rhoNew`, `Mmat`, `Y`, `alphaMat`, `betaMat`, `coefWork`, `rhsWork`, `Rqrcp` (all `s x s`), `Tmp`
(`s x n`, shared GEMM-output scratch -- distinct from the public `T` parameter, named `Tmp` specifically
to avoid confusion with it), `Pqrcp` (`Pivot(s)`), `uQrcp` (`fProxyN(s)`), `thr` (`fProxyN(s)`), and
`rowIn`/`rowOut` (`fProxyN(n)`, only if `!M.IsIdentity`) -- same disposal-at-cleanup discipline as `bcg`.

## 10. Edge cases

- Initial `R` already converged (`lastConverged == s` at setup) -> `Converged`, `iterations = 0`.
- `BlockFrobDot(rhoNew,rhoNew) == 0` or NaN at S2, at any `k` (including `k=0`) -> `Breakdown`.
- `BlockSolveGeneral` reports `RankDeficient` (rank `< s`) at S3 or S10 -> `Breakdown` (this is the
  **defined, in-scope** behavior for a rank-deficient block -- see Section 14, full continuation/deflation
  through a rank loss is explicitly out of scope for this first-cut block-BiCGSTAB, matching
  `docs/dev/spec-block-krylov.md`'s "first cut: keep-all-columns" guidance).
- `tT <= 0` (NaN-safe) or `omega == 0`/NaN at S15 -> `Breakdown`.
- `s == 1`: degenerates toward scalar `biCGStab` through the same code path (`BlockSolveGeneral` on a
  `1x1` system is a trivial divide). No special-casing required; a smoke test is welcome but not
  mandatory.
- Two exactly identical RHS columns: expected to trigger `Breakdown` at S3 or S10 on an early iteration
  (identical rows make `rho`/`Mmat` exactly rank-deficient) -- this is the "graceful handling of a
  rank-deficient RHS block" requirement: finite, non-NaN `X` and an honest `Breakdown` status, NOT a
  successful degraded solve (Section 14 scopes that out).
- Warm-started `X` (nonzero on entry): must work -- nothing in Section 4 assumes `X` starts at zero.
- `maxIter` exhausted with columns still unconverged: `status = MaxIterations`, `converged < s`, `X` holds
  the best iterate reached (`lastConverged`/`lastMaxRnorm` as tracked through S13/S18).

## 11. Tests -- new file `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BlockBiCGStabTests.fProxy.cs`

Mirror `BlockCGTests.fProxy.cs`'s structure exactly: one `[BurstCompile(CompileSynchronously = true)]
IJob` struct with a `TestType` enum switch, every scenario built and asserted **inside** `Execute()`,
each `[Test]` method doing `new ...Job { Type = ... }.Run()`. `Assert.IsTrue(bool)` / `Assert.AreEqual`
only -- **never** the string-message overload (BC1071 -> silent Mono fallback).

New private helper needed (nonsymmetric analog of `BuildDenseSPD` -- do NOT form `M^T M`):
```
static fProxyMxN BuildDenseNonSym(ref Arena arena, int dim, uint seed)
{
    var A = arena.fProxyRandomMat(dim, dim, (fProxy)(-1f), (fProxy)1f, seed);
    for (int d = 0; d < dim; d++) A[d, d] += dim;   // diagonally dominant -> nonsingular, nonsymmetric
    return A;
}
```

Required test cases:

1. **`MatchesScalarBiCGStabPerColumn`** -- `Krylov.bbiCGStab` on a `BuildDenseNonSym` system (`s >= 3`)
   matches `s` independent scalar `Krylov.biCGStab` solves per column to `tol`-scaled tolerance;
   `info.Solved` and `info.converged == s`. This is the primary correctness oracle for Section 6's open
   risk.
2. **`KnownSolutionRecovered`** -- known `Xk` (`arena.fProxyRandomMat`), `B = A . Xk` via `ApplyBlock` on
   a `BuildDenseNonSym` operator, recover `Xk` to tolerance.
3. **`IdentityFoldBitIdentical`** -- `bbiCGStab<TOp, fProxyIdentityPreconditioner>` (explicit identity)
   produces **bit-identical** `X`/`iterations`/`status` to the unpreconditioned `bbiCGStab<TOp>` overload
   on the same fixed-seed nonsymmetric system (exact equality, no tolerance).
4. **`RankDeficientRHSBlockBreaksDownGracefully`** -- two RHS columns forced exactly identical (as
   `BlockCGTests.RankDeficientBlockDeflates`, but on `BuildDenseNonSym`). Assert: every entry of `X` is
   finite (no NaN/Inf); `info.status == IterativeSolveStatus.Breakdown` (Section 10's defined behavior --
   do NOT assert `Solved`).
5. **`PreconditionedMatchesScalar`** -- BSR (`arena.fProxyRandomSparse`, nonsymmetric diag-dominant) +
   `arena.fProxyRestrictedSchwarz`, matches per-column scalar `Krylov.biCGStab(in A, in M, in bj, ...)`
   (mirrors `SchwarzPreconditionerTests.RasResidualOk`'s construction).
6. **`MaxIterBudgetHonestStatus`** -- a deliberately tiny `maxIter` (e.g. 2) on a system that needs more:
   `status == IterativeSolveStatus.MaxIterations`, `converged < s`, `X` finite (no NaN/Inf), no throw.
7. **`JobSafeThroughRun`** -- satisfied by construction (every test above runs via `IJob.Run()`); no
   extra test needed, but confirm in review that the core never reassigns which physical buffer a
   `ref fProxyMxN` parameter (`R`, `P`, `V`, `T`) points to -- no ping-pong/`SwapMat`, so no
   `RestoreBufferIdentity`-style IJob-struct-copy hazard to guard against.

All tests use `Consts.fProxySqrtEps`-scale tolerances and the same `Tol()`/`Row`/`DenseToBSR1x1`-style
private helpers as `BlockCGTests.fProxy.cs` (copy/adapt, do not import cross-file).

## 12. Implementation checklist (ordered)

1. Add `BlockSolveGeneral`, `BlockFrobDot`, `BlockScaleInPlace` private helpers to
   `Krylov.Block.fProxy.cs` (Section 7).
2. Implement the generic core `bbiCGStab<TOp, TPre>` per Section 4 exactly (setup 4.2, loop 4.3),
   including the `lastConverged`/`lastMaxRnorm` tracking discipline spelled out at each breakdown point.
3. Add the 7 forwarding/convenience overloads per Section 9.
4. Regenerate (`Tools/regen.ps1`) and confirm float+double both compile clean.
5. Write `BlockBiCGStabTests.fProxy.cs` (Section 11, all 7 scenarios).
6. Run the full suite headlessly; confirm the exact line `Result=Passed total=N passed=N failed=0`
   (never pipe through `| tail`).
7. If test #1 or #2 (Section 11) fail to converge, try the Section 6 mitigation (swap S4's
   multiplication order) before assuming a different bug; once resolved, add a `## Krylov.bbiCGStab`
   `DEVLOG.md` entry (dated, newest-first, per `CLAUDE.md`'s format) recording which order was
   empirically correct. Do **not** put this in code comments.

## 13. Acceptance criteria

- `Krylov.bbiCGStab` exists in `OP/Krylov.Block.fProxy.cs` with the 8-overload ladder of Section 9,
  generated cleanly for both `float` and `double`.
- Existing `Krylov.bcg` block overloads and all reused private helpers (`BlockGram`, `BlockCTV`,
  `BlockAdd`, `BlockZplusT`, `CopyBlock`, `CopyMat`, `BlockApplyPre`, `CountConverged`) are **unmodified**
  -- no signature or body changes.
- All 7 tests in `BlockBiCGStabTests.fProxy.cs` (Section 11) exist and pass, including
  `MatchesScalarBiCGStabPerColumn` and `KnownSolutionRecovered` on a genuinely nonsymmetric system.
- `RankDeficientRHSBlockBreaksDownGracefully` asserts finite `X` AND `status == Breakdown` (not
  `Solved`).
- `IdentityFoldBitIdentical` passes with **exact** (non-tolerance) equality.
- Full project test suite green: the literal line `Result=Passed total=N passed=N failed=0` from the
  headless test run, `N` including the 7 new tests, `failed=0`.
- No edits to `README.md`. No edits to anything under `Assets/LinearAlgebra/Source/` (generated output --
  regenerate instead). No edits to `Pivot/Pivot.cs` or `Pivot/Pivot.Operations.cs`.
- If the Section 6 multiplication-order swap was needed, a `## Krylov.bbiCGStab` entry exists in
  `OP/DEVLOG.md` recording it.

## 14. Out of scope (do not do these in this task)

- Restart-on-breakdown or any partial-rank continuation strategy for a rank-deficient `rho`/`Mmat`
  (Section 10) -- this first-cut solver reports `Breakdown` cleanly instead; a deflating/restarting
  variant (the `bcgrq`-style treatment, applied to BiCGSTAB) is a separate future task, not this one.
- Column locking / per-column active-width tracking (the `bcgrq`-style `sLive`/`Live` mechanism) -- this
  solver is "first cut: keep-all-columns" per `docs/dev/spec-block-krylov.md`'s explicit guidance; every
  live column receives an update every iteration until ALL converge together.
- Dedicated per-preconditioner concrete overloads (`fProxyBlockJacobi`/`SSOR`/`IC0`/`FSAI`/`Chebyshev`/
  `AdditiveSchwarz`/`RestrictedSchwarz`-specific rungs) mirroring the scalar file's -- only the generic
  `<TPre>` rungs (Section 9, rungs 5 and 8).
- Any benchmark file/row (`BlockCGSparseBenchmark` or similar) -- not requested for this task.
- Any change to `BlockSolveInfo`, `RankInfo`, `DirectSolveStatus`, or `IterativeSolveStatus`.
- Resolving the `Pivot` "Arena dependency?" TODO (priority-backlog item 2) -- `bbiCGStab` uses `Pivot`
  exactly as it exists today.
- Renaming `bbiCGStab` to the suggested `bBiCGStab` (Section 0) -- implement under the working name
  given; a rename is a trivial follow-up if the project owner picks the alternative later.
- Block-MINRES / block-GMRES / `bcgrq` itself -- untouched, separate tasks.
- SVD, least squares, optimizers, sparse-matrix work (beyond BSR reuse), View/Slice -- unrelated to this
  task.
