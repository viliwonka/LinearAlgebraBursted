# Krylov DRY-extraction survey (Task #57, step 1)

Read-only inventory of repeated code across all Krylov solver templates under
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/`, feeding the pre-optimization extraction pass
(task #57) and the SIMD/optimization pass (task #58). Every claim is cited `file:line`. All paths
are relative to `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/` unless stated.

Scope surveyed: 13 single-RHS solvers (CG, FCG, MINRES, MINRESQLP, BiCGStab, GMRES, FGMRES, IDR,
TFQMR, CRAIG, CRAIGMR, LSQR, LSMR, GCRODR) + 14 block solvers (Block.CG, BCGrQ, BFBCG, BiCGStab,
GMRES, FGMRES, MINRES, IDR, TFQMR, LSMR, CGLS, CRAIG, CRAIGMR, GCRODR) + the shared files
(Block.Common, Lstsq.Common, Guards) + existing kernels (UnsafeOP, OP.Component, Blas.*).

---

## 0. Existing shared infrastructure (what to REUSE before writing anything new)

### 0.1 Vector BLAS-1 (single-RHS elementwise/reductions) — already SIMD, already routed
`OP.Component.fProxy.cs` extension methods on any `IUnsafefProxyArray`, all forwarding to `UnsafeOP`:
- `zeroInPlace` (16), `fillInPlace` (25), `addInPlace(scalar)` (33), `mulInPlace` (41),
  `divInPlace` (49), `addInPlace(T)` (67), `subInPlace` (76), `addScaledInPlace` = axpy (85),
  `scaleAddInPlace` = aypx (94).

### 0.2 Fused Krylov reduction kernels — already written, already SIMD (fProxyW + fProxy4)
`UnsafeOP.fProxy.cs` (exposed via `Blas.*` wrappers used by the solvers):
- `vecDot` (242), `vecDotRange` (287) — the canonical reduction tree every fused kernel mirrors.
- `axpyNormSq` (1782): `y += a*x; return dot(y,y)`.
- `xpayNormSq` (1847): `y = a*y + x; return dot(y,y)` — the Golub-Kahan workhorse (used by
  lsqr/lsmr/craig/craigmr).
- `updateXR` (1914): `x += a*p; r -= a*q; return dot(r,r)` — the CG twin-update, used by cg (via
  `Blas.updateXR`, CG.fProxy.cs:207).
- `scaledCopy` (1980): `y = a*x` (reciprocal-multiply copy).
- `combine3` (1991): `w = s*(v + a*w1 + b*w2)` — the MINRES 3-term direction update (MINRES:186).
- `addSquares` (2000), `jacobiRotate` (2013), `francisRow3/2` (2030/2045) — the last two are
  the SIMD plane-rotation butterflies (relevant to the Givens clusters, §B/§H).

Contract for all of the above: the fused reduction half is **bit-identical** to
"plain update kernel then a separate `vecDot`" (comment block at UnsafeOP.fProxy.cs:1772-1777).
This bit-identity contract is the acceptance bar for any new fused kernel.

### 0.3 Distinct-buffer guard
`Krylov.Guards.cs:14` `RequireDistinctBuffers(who, long* ptrs, count)` (singular partial).

### 0.4 Block (multi-RHS) shared helpers — `Krylov.Block.Common.fProxy.cs`
`BlockGram` (18), `BlockCTV` (31), `BlockAdd` (35), `BlockZplusT` (43), `BlockSolveSPD` (54),
`CountConverged` (79), `BlockApplyPre` (94), `CopyBlock` (106), `CopyMat` (112), `View` (120),
`RowsView` (124), `RectView` (128), `BlockScatterAddRows` (132), `LockConvergedRows` (147),
`LQRPRank` (167), `FactorLiveResidual` (185), `FactorLiveSearch` (214), `FactorGramOnce` (232).

### 0.5 Least-squares shared tails — `Krylov.Lstsq.Common.fProxy.cs`
`lstsqResidual` (25), `LstsqInfoTracked` (58), `JacobiFinish` (101).

---

## 1. DUPLICATION INVENTORY (clusters, newest/biggest first)

### Cluster A — Single-RHS convenience-overload ladder  ★ largest LOC, lowest risk
Every single-RHS solver carries a fan of near-identical forwarding overloads: `(dense A)` →
`fProxyDenseOperator`, `(BSR A)` → `fProxyBSROperator`, `(A, M, ...)` preconditioned rungs, an
arena-allocating rung that news up N `fProxyTempVec` scratch and calls the zero-alloc rung, and a
defaults rung (`maxIter = A.M_Rows`, `tol = Consts.fProxySqrtEps`). Each rung is 3-15 mechanical
lines; the ladder dwarfs the actual solver.

Method counts (public forwarders per file): CG **28**, MINRES **28**, LSQR/LSMR **15**, BiCGStab
**14**, MINRESQLP 13, IDR/TFQMR 10, GMRES/FGMRES/CRAIG/CRAIGMR/GCRODR 8, FCG 3.

Worst offender is the **preconditioner ladder**: CG repeats the identical 3-rung block
(zero-alloc forward / arena-alloc / defaults) once per preconditioner type — BlockJacobi, SSOR,
IC0, FSAI, Chebyshev, AdditiveSchwarz — at CG.fProxy.cs:278-306, 314-342, 350-378, 386-414,
422-450, 459-487. That is **six copies** of the same ~30-line shape, differing only in the `M`
type name and the XML doc. The arena-allocating rung itself (`r/p/Ap/z = b.fProxyTempVec(...)`
then forward) is byte-identical across all six (e.g. 289-297 vs 325-333 vs 361-369 …).

Same shape, dense/BSR/defaults rungs: CG 39-100, MINRES 243-289+, GMRES 199-221, and every other
single-RHS file's tail.

- Copies: ~14 files, dozens of rungs. **Highest raw-line-count duplication in the whole survey.**

### Cluster B — GMRES / FGMRES scalar Arnoldi + Givens + Hessenberg back-solve  ★ near-verbatim
`Krylov.GMRES.fProxy.cs` and `Krylov.FGMRES.fProxy.cs` share the inner Arnoldi cycle almost
character-for-character. The only genuine difference is the preconditioned-basis storage: GMRES
applies `M⁻¹` into a single reused `zt` and re-applies `M⁻¹` once at the commit (GMRES:157-175);
FGMRES stores every `z_j` in `Z[j]` for a flexible commit (FGMRES:107-109, 168-183).

Byte-identical (or trivially so) sub-blocks, GMRES ↔ FGMRES:
- Modified Gram-Schmidt loop: GMRES:102-118 ↔ FGMRES:112-128.
- Apply previous Givens rotations to column j: GMRES:120-126 ↔ FGMRES:130-136.
- New Givens rotation zeroing `H[j+1,j]` + rotate rhs `g`: GMRES:128-145 ↔ FGMRES:138-150.
- Hessenberg back-substitution: GMRES:148-154 ↔ FGMRES:158-164.
- `bb==0` init + `bnorm`/`thresh` + restart residual `v0 = b - A x`, `beta=||v0||`:
  GMRES:45-85 ↔ FGMRES:45-92.

`Krylov.GCRODR.fProxy.cs` carries a **third copy** of the same MGS + Givens + back-solve
machinery: MGS at GCRODR:180-189, previous-Givens at 199-206, new-Givens at 207-215,
back-substitute at 229+ (comment "SAME Hessenberg/Givens least-squares machinery" at GCRODR:21).

- Copies: **3** (gmres, fgmres, gcrodr) of the MGS/Givens/back-solve core.

### Cluster C — Single-RHS init + threshold + verify-at-exit (true-residual recompute)
Two tightly-repeated idioms across the SPD/nonsymmetric single-RHS solvers.

(C1) **Init + trivial-b guard + relative threshold**:
`bb = Blas.dot(b,b); if (bb == 0) { x.CopyFrom(b); return Converged,0,0 }`; then
`r = b - A x` (Apply into scratch, CopyFrom b, addScaledInPlace(-1)); then
`threshold = tol*tol*bb`. Appears at:
CG:160-173, MINRES:73-86, MINRESQLP:70-74, BiCGStab:62-75, GMRES:45-52, FGMRES:45-53,
FCG:67-80, IDR:45-51, TFQMR:73-78, GCRODR:62-66. The least-squares variant uses
`atbSq = ||Aᵀb||²` as the scale instead of `||b||²`: LSQR:59-69, LSMR:68-75, CRAIG:61,
CRAIGMR:64. (10 `||b||²` copies + 4 `||Aᵀb||²` copies.)

(C2) **Verify-at-exit** — when the *tracked* residual estimate crosses threshold, recompute a
FRESH true residual `A.Apply(x) → r = b - Ax`, re-check, and only then return Converged (else keep
iterating). Confirmed copies (comment "Verify-at-exit"):
CG:210-217, MINRES:192-204 (and a second fresh recompute for preconditioned MaxIterations at
MINRES:214-219), BiCGStab:125-… and 184-…, IDR:170-… and 227-…, MINRESQLP:373-388 (final fresh
true residual + honesty guard). TFQMR documents the same intent (TFQMR:15,32) via its per-half-step
recompute.

- Copies: C1 ~14, C2 ~6 solvers (several with two recompute sites each).

### Cluster D — Scalar Golub-Kahan bidiagonalization  (lsqr / lsmr / craig / craigmr)
Identical bidiag recurrence built on the fused `Blas.xpayNormSq`:
- init: `u = b - A x0; beta=||u||; u/=beta; v = Aᵀu; alpha=||v||; v/=alpha`
  (LSQR:71-94, LSMR:77-…, CRAIG:~68, CRAIGMR:71-84).
- step: `u = A v - alpha u; beta=||u||` then `v = Aᵀu - beta v; alpha=||v||`, each folded into one
  `Blas.xpayNormSq` pass (LSQR:116-125, LSMR similar, CRAIG:80-116, CRAIGMR:99-154).

- Copies: **4**. Landmine: **step ordering differs** — lsqr/lsmr do u-step then v-step; CRAIG does
  the v-step first (CRAIG:80-82 before 100-101); CRAIGMR's init uses `Blas.dot` not the fused form
  (CRAIGMR:77). A merged helper must expose both orderings (lower- vs upper-bidiagonal) or be a
  single-step primitive the caller sequences.

### Cluster E — Block-solver skeleton (X=0 init, per-column threshold, final maxRnorm, mass Dispose)
Every block solver reuses the same top-and-tail:
- (E1) `X = 0` init double-loop: Block.CGLS:42-43, Block.CRAIG, Block.CRAIGMR, Block.LSMR:147-148
  (block-LS group; warm-started block solvers skip it).
- (E2) per-column threshold `thr[j] = tol*tol*<scale[j]>`: Block.BCGrQ, BFBCG, BiCGStab, CG, CGLS
  (69), FGMRES, GCRODR, GMRES, IDR, MINRES — **10 files**.
- (E3) **final maxRnorm cleanup loop** — `double maxr=0; for each row j { double rr=0; for c
  rr += d*d; maxr = max(maxr, sqrt(rr)) }` then build the `BlockSolveInfo` and Dispose everything.
  Present in **all 13** block files (Block.CGLS:113-133, Block.LSMR:358-388, Block.CRAIG:164-…,
  Block.CRAIGMR:211-…, and the rest). The residual source differs (raw `R` block for CGLS:121 vs a
  fresh `B - A X` recompute for the LS solvers, Block.LSMR:360-372 / Block.CRAIG:171 /
  Block.CRAIGMR:218) — so E3 splits into "residual already in R" vs "recompute R" variants.
- (E4) the giant `Dispose()` cascade (Block.LSMR:375-385, Block.CGLS:128-130, …) — mechanical,
  one call per Temp buffer.

- Copies: 13 (E3), 10 (E2), 4 (E1).

### Cluster F — Block helpers hosted in a solver file but shared cross-file (misplaced, not dup)
These are already single-definition (no code duplication) but **live in a specific solver rather
than Block.Common**, so the dependency graph reads backwards (a sibling `#include`-style coupling
that codegen tolerates because it's one partial class). Moving them to Block.Common is pure
housekeeping that unblocks clean review of the rest:
- Defined in `Krylov.Block.LSMR.fProxy.cs`, used by CGLS/CRAIG/CRAIGMR/LSMR:
  `BlockApplyOp` (17), `BlockApplyOpT` (31), `TriNearSingular` (48), `ExtractBlockTranspose` (64),
  `ExtractBlockAt` (72), `WriteBlockAt` (82), `TransposeSmall` (90), `BlockSolveGeneralWide` (101).
- Defined in `Krylov.Block.BiCGStab.fProxy.cs`, used widely:
  `BlockCrossGram` (52, used by BiCGStab/FGMRES/GCRODR/GMRES/IDR), `BlockSolveGeneral` (58, used by
  BiCGStab/IDR), `BlockFrobDot` (69, used by BiCGStab/CRAIG/CRAIGMR/IDR/LSMR/TFQMR),
  `BlockScaleInPlace` (79, used by BiCGStab/CRAIG/CRAIGMR/IDR/LSMR).
- Defined in `Krylov.Block.GMRES.fProxy.cs`, used by FGMRES/GCRODR/GMRES:
  `StoreBlockAt`, `ExtractRowsAt`, `ZeroPrefix`.

### Cluster G — Block Golub-Kahan bidiag + block-Givens 2s×2s QR  (blsmr / bcraig / bcraigmr)
The block generalization of Cluster D: s×s LQ factors of the residual/`AᵀU` blocks each step, and
LSMR's two Givens stages lifted to a **block-QR of a stacked 2s×s pair, zero-padded to 2s×2s**
(Block.LSMR:290-306 — `WriteBlockAt` the pair into `Mpad`, `QR.decomp`, then `ExtractBlockAt` /
`ExtractBlockTranspose` the four s×s sub-blocks a/b/c/d). `LQ.decomp`/`BlockApplyOp` bidiag-step
density: Block.LSMR 13 calls, Block.CRAIG 12, Block.CRAIGMR 11 (vs CGLS's 3 — CGLS is normal-eq CG,
not bidiag). The `TriNearSingular` breakdown guard fires on every LQ factor (Block.LSMR:258,271,300).

- Copies: **3** (blsmr, bcraig, bcraigmr) of the block-bidiag step + 2s block-Givens.

### Cluster H — Block Arnoldi + dense-QR least-squares re-solve + Pythagorean check  ★ near-verbatim
`Krylov.Block.GMRES / FGMRES / GCRODR` share the entire inner block cycle:
- block matvec via `ApplyBlock` (+ `BlockApplyPre` under a real M): Block.GMRES:176-187.
- modified block Gram-Schmidt with one unconditional reorthogonalization (MGS2), storing s×s
  `Hij` blocks via `BlockCrossGram` + `StoreBlockAt` + `BlockCTV`/`BlockAdd` back-subtraction:
  Block.GMRES:189-203.
- deflating thin-LQ of the residual block (`LQRP.decomp` → rank via `LQRPRank`): Block.GMRES:205-224.
- **periodic dense re-QR least-squares solve**: `QR.decompInPlace(HQ,Rls)` then
  `QR.decompSolve(HQ,Rls,Gactive,Yv)` — Block.GMRES:238/241, Block.FGMRES:211/214,
  Block.GCRODR:329/332 (essentially identical).
- **per-column Pythagorean LS-residual convergence check** (`resid2 = max(0, gg - qq)` vs
  `thr[c]`): Block.GMRES:246-255, Block.FGMRES:219-229, Block.GCRODR:345-355 — byte-identical.

- Copies: **3**. GCRODR adds a recycle-subspace step (extra `QR.decompInPlace` at GCRODR:542) on
  top of the shared core.

### Cluster I — Distinct-buffer guard: helper vs inline (inconsistency, not duplication)
Most solvers call the shared `RequireDistinctBuffers` (MINRES:70, MINRESQLP:67, BiCGStab:59,
TFQMR:70, CRAIG:55, CRAIGMR:57, LSQR:55, LSMR:61). Two still hand-expand an OR-chain of pointer
comparisons: **CG** (CG.fProxy.cs:142-158) and **FCG** (FCG.fProxy.cs:~64). MINRES's own comment
(MINRES:13) flags the OR-chain as the thing `RequireDistinctBuffers` replaced — CG/FCG were missed.

---

## 2. Per-cluster: reuse existing vs new helper, and the merge landmines

| # | Cluster | Verdict | Landmines |
|---|---------|---------|-----------|
| A | Overload ladder | **New codegen mechanism** (not a runtime helper). Best fix is a codegen macro / template-expansion that stamps the {dense, BSR, precond×N, defaults} rungs from one declaration, OR a `//+choose[...]` list over preconditioner types. A runtime generic can't erase the per-M-type overloads (Burst needs concrete specializations). | The precond rungs differ only by `M` type + which BSR preconditioners are *valid* for that solver (CG excludes RAS — CG:457 note; biCGStab includes it). Any generator must take a per-solver preconditioner allow-list, not a fixed set. Defaults differ (`A.M_Rows` vs `min(30,N)` restart for GMRES:205). |
| B | GMRES/FGMRES/GCRODR MGS+Givens+backsolve | **New shared helpers**: `ArnoldiMGSStep` (dot+axpy loop + normalize), `GivensApplyAndGenerate` (rotate H column j + new rotation + rhs), `HessenbergBackSolve`. All three are pure scalar/small-matrix; extract verbatim. | GMRES reuses one `zt`; FGMRES stores `Z[j]`; GCRODR prepends recycle columns to the Arnoldi basis (its `H` has an extra leading block). Helper must operate on the H/g/cs/sn arrays, leaving basis management to the caller. |
| C | Init + verify-at-exit | **New shared helpers**: `KrylovInit` (bb + trivial-b return + `r=b-Ax` + threshold) and `VerifyTrueResidual(A, b, x, ref rScratch) → rr`. | Scale differs (`||b||²` vs `||Aᵀb||²`); return type differs (`SolveInfo` vs `LstsqInfo` — the LS group can't share the trivial-b return path). Verify uses whatever scratch is idle at that point (MINRES reuses `y`+`v`, IDR reuses `V`/`Q`) — helper takes explicit scratch refs. C2 must **fall through and keep iterating** on a failed verify (not return) — easy to get wrong. |
| D | Scalar Golub-Kahan | **New shared helper**: `GolubKahanStep(A, ref u, ref v, ref alpha, ref beta, ref tmpM, ref tmpN)` built on the existing `Blas.xpayNormSq`. | Step ordering (lower vs upper bidiag) differs lsqr/lsmr vs craig; craigmr init uses `Blas.dot`. Make it a **single half-step** primitive the caller sequences, or two named entry points. Damping (lsqr/lsmr) folds AFTER the step and stays in the caller. |
| E | Block skeleton | **New shared helpers** in Block.Common: `BlockZeroInit(ref X)`, `BlockPerColThreshold(scaleGram, tol, ref thr)`, and two `BlockFinishInfo` tails — one reading the live `R`, one recomputing `B - A X` via `BlockApplyOp`. | E3's residual source differs (R vs recompute). The `converged` count semantics: joint-Frobenius solvers set `converged = s or 0` (Block.LSMR:373), per-column solvers count live columns. Keep those in the caller; share only the maxRnorm loop + struct assembly. Dispose cascade (E4) is not worth abstracting (each solver's buffer set differs). |
| F | Misplaced hosts | **Move to Block.Common** (housekeeping). No behavior change. | Pure relocation; codegen reads all `*.cs` in the folder so the partial-class members resolve identically. Do this FIRST — it makes G/H review tractable. |
| G | Block bidiag + 2s Givens | **New shared helper**: `BlockGolubKahanStep` (the LQ-of-residual / LQ-of-AᵀU pair + `TriNearSingular` guards) and `BlockGivens2s` (the `Mpad`/`QR.decomp`/extract-4-subblocks pattern, Block.LSMR:290-306). | The four extracted sub-blocks are consumed differently per solver; `BlockGivens2s` should return the four s×s blocks (a,b,c,d) + `alphabark`, caller does the recurrence. Ridge/rank guards must stay (Block.LSMR:300,313). |
| H | Block Arnoldi + dense-QR LS | **New shared helper**: `BlockArnoldiMGS2Step` + `BlockLSResolveAndCheck` (the `QR.decompInPlace`/`decompSolve`/Pythagorean-check block, Block.GMRES:229-256). | GCRODR's leading recycle block changes `off[]`/`H` offsets; helper must take the offset arrays. FGMRES stores `Z`; GMRES reuses `Zt`. Convergence check is identical — extract that first (safest). |
| I | Guard helper vs inline | **Reuse** `RequireDistinctBuffers`; convert CG:142-158 and FCG to it. | Trivial. CG passes `z` conditionally (only for a real M) — mirror MINRES:64-69's `count = IsIdentity ? 9 : 10` idiom. |

---

## 3. SIMD / vectorization opportunities (feeds task #58)

Most single-RHS inner loops are **already routed through the SIMD-fused kernels** (§0.2): CG's
twin update is `Blas.updateXR` (CG:207), the bidiag solvers use `Blas.xpayNormSq`, MINRES uses
`Blas.combine3` + `Blas.scaledCopy`. So for those, extraction is about DRY, not new SIMD. The
open SIMD targets are the block and GMRES-family scalar loops that still hand-roll element loops:

- **Cluster B/H Modified Gram-Schmidt** (GMRES:102-118, Block.GMRES:189-203): a `dot` (reduction)
  + `axpy` (elementwise) per basis vector. The `dot`s already hit `Blas.dot`/`BlockCrossGram`; the
  raw `v0[i] *= invBeta` normalize loops (GMRES:83,117; FGMRES:90,127) are scalar and should route
  to `Blas.scaledCopy`/`mulInPlace` (SIMD, elementwise map). **Good fProxyW candidate.**
- **Cluster B Givens column update** (GMRES:120-141): the "apply previous rotations" loop is a
  sequential scalar recurrence over H's column — this is exactly the `jacobiRotate`/`francisRow*`
  butterfly shape (§0.2) but on a single column; low width, marginal SIMD value. Leave scalar.
- **Cluster E3 maxRnorm loop** (Block.LSMR:362-372, all block files): per-row `Σ d*d` reduction —
  a textbook `vecDot`/`fProxyW` reduction. Currently hand-rolled `double` accumulation. Routing to
  a widened reduction is a **clean fProxyW win** and centralizing it in one helper does it once.
- **Cluster G `BlockFrobDot`** (Block.BiCGStab:69) and **`BlockScaleInPlace`** (79): whole-block
  contiguous reduction / scalar-map over `M_Rows*N_Cols`. `BlockFrobDot` is a `vecDot` over the
  flat buffer (reduction → fProxyW); `BlockScaleInPlace` is `mulInPlace` (already SIMD if routed).
  Both currently scalar `for (long i…)`. **fProxyW candidates**, and they're hot (called per iter).
- **Cluster D/G block-marshalling** (`ExtractBlockTranspose`, `TransposeSmall`, `WriteBlockAt`,
  `BlockApplyOp` row copies, Block.LSMR:24-26,38-40,64-95): gather/scatter + transpose of small
  s×s / s×n blocks. Transposes are strided (poor SIMD); the contiguous row copies inside
  `BlockApplyOp` could use `UnsafeUtility.MemCpy`. Low priority — s is small.
- **Cluster C init `r = b - A x`** (CopyFrom + addScaledInPlace): already two SIMD kernels; a fused
  `sub-into` (`r = b - Ax` in one pass) is a possible new kernel but marginal (runs once).

Net: the **highest-value new SIMD** is centralizing `BlockFrobDot`, the E3 per-row residual
reduction, and the GMRES/block normalize maps — all reductions/elementwise maps that fit fProxyW.

---

## 4. PRIORITIZED extraction plan

Order = (duplication removed) × (safety) — mechanical/verbatim first, numerics-sensitive last.

**P0 — Housekeeping, zero behavior change (do first, unblocks the rest)**
1. **Cluster F**: relocate the shared block helpers from Block.LSMR / Block.BiCGStab / Block.GMRES
   into `Krylov.Block.Common.fProxy.cs`. Pure move; regen + suite must be byte-identical.
2. **Cluster I**: convert CG:142-158 and FCG's inline OR-chain to `RequireDistinctBuffers`
   (mirror MINRES:64-70). Tiny, verifiable.

**P1 — Big mechanical extracts, near-verbatim copies**
3. **Cluster B**: extract `ArnoldiMGSStep` + `GivensApplyAndGenerate` + `HessenbergBackSolve`;
   retarget gmres, fgmres, gcrodr. Copies are already character-identical, so the diff is safe;
   bit-identity is the acceptance bar.
4. **Cluster H**: extract `BlockLSResolveAndCheck` (the dense-QR + Pythagorean check,
   Block.GMRES:229-256) first (identical across bgmres/bfgmres/bgcrodr), then
   `BlockArnoldiMGS2Step`. Retarget the 3 block-Arnoldi solvers.
5. **Cluster E3**: extract the `BlockFinishInfo` maxRnorm tail (two variants: R-in-hand vs
   recompute). Touches all 13 block files but each call site shrinks to one line.

**P2 — Structured extracts with a numeric contract to preserve**
6. **Cluster C**: `KrylovInit` + `VerifyTrueResidual`. Watch the fall-through-on-failed-verify
   semantics (C2) and the `SolveInfo` vs `LstsqInfo` split. Verify per solver with residual-based
   checks (not just pass/fail) — a wrong verify silently reports false Converged.
7. **Cluster D**: `GolubKahanStep` half-step primitive over `Blas.xpayNormSq`; retarget
   lsqr/lsmr/craig/craigmr. Preserve step ordering per solver (the landmine).
8. **Cluster G**: `BlockGolubKahanStep` + `BlockGivens2s`; retarget blsmr/bcraig/bcraigmr. Keep
   every `TriNearSingular`/rank guard exactly where it is.

**P3 — Codegen-level (separate, larger design)**
9. **Cluster A**: design a codegen expansion for the overload ladder (per-solver preconditioner
   allow-list + defaults). Biggest LOC win but it's a template-generation change, not a helper
   extract — schedule after P0-P2 have proven the runtime helpers stable. **This is where the
   line-count actually collapses.**

**Leave alone**
- The Dispose cascades (E4): each solver's buffer set is genuinely different; a shared disposer
  would take an ever-changing arg list. Not worth it.
- The scalar Givens "apply previous rotations" recurrence as a SIMD target (§3): sequential, low
  width. DRY-extract it (P1.3) but don't vectorize.
- FCG is only 170 lines / 3 overloads — smallest ROI on the ladder; fold it in with CG's changes
  opportunistically, don't special-case it.

### SIMD follow-ups for task #58 (from §3), independent of the DRY extracts
- Route `BlockFrobDot` (Block.BiCGStab:69) and the E3 per-row residual reduction to a fProxyW
  reduction (hot, clean win).
- Route the GMRES/FGMRES/block normalize loops (GMRES:83,117; FGMRES:90,127) and
  `BlockScaleInPlace` to the existing `Blas.scaledCopy`/`mulInPlace` SIMD maps.
