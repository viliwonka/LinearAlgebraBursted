# Spec: Solver/Decomposition API Rework

Status: APPROVED 2026-07-05 — all open questions resolved (see bottom). Implementation may begin.
Scope: dense direct-solver surface (LU, Cholesky, QR, QRCP, LQ, + Solvers primitives).
Pre-release breaking change; no compatibility shims (see OQ-9).

## Motivation

The current surface mixes four contracts under inconsistent names: `qrDirectSolve` destroys
A and b, `qrcpDirectSolve` preserves both, `choleskySolve` is factor-consuming in one
overload and a fused driver in another, `luDecomposition` silently destroys A, and
`Solvers.solveQR(A,b,x)` forwards to a destructive kernel under a factor-consuming name.
Parameter names lie in both directions (`solveUpperTriangular`'s `x` enters holding b;
`qrDecomposition`'s `Q` enters holding A). This rework makes every name state its contract.

## Naming scheme

**Bare methods on algorithm classes** (precedent: `Blas.dot`, `Norms.L1`, `Solvers.cg`).
The class names the algorithm; methods name the operation. Methods are lowerCamel.

Four tokens, one meaning each:

| Token | Meaning |
|---|---|
| `decomp` | factor; `in A` preserved; factors into caller buffers |
| `decompInPlace` | factor into A's own storage; A destroyed (becomes a factor) |
| `decompSolve` | solve from existing factors; factors read-only; solve-many tier |
| `solveInPlace` | one-shot solve, fastest path for the algorithm, destructive |

Rules:
- **No safe one-shot solves.** Never pay for safety: nothing in the API copies a matrix
  to protect the caller. Want A preserved → `decomp` + `decompSolve`, or copy explicitly.
  (Where safety is free — LQ/SVD one-shots only *read* A — it is kept, not bought.)
- **Param transformation names**: a `ref` param whose exit state is a *documented, usable
  value* is named `in_to_out`, case-faithful: `A_to_Q`, `A_to_LU`, `A_to_L`, `b_to_x`.
  A `ref` param whose exit state is scratch/undefined keeps its plain name and the XML doc
  says "destroyed; contents undefined after return" (QR.solveInPlace's `A`, `b`).
- **Output-only params are uninit-safe** (see Contracts).

## Target API

`fProxy` = float/double via codegen; all templates in `CodeGen/TemplateSource/OP/`.

### LU
```
decomp        (in A, ref L, ref U, ref Pivot P)            // NEW BEHAVIOR: A preserved (today's luDecomposition destroys A-as-U)
decompInPlace (ref A_to_LU, ref Pivot P)                   // = luDecompositionInPlace (compact storage), rename only
decompSolve   (ref LU, in Pivot P, ref b_to_x)             // = luSolve(ref LU,...), rename + param rename
decompSolve   (ref L, ref U, in Pivot P, ref b_to_x)       // = luSolve(ref L, ref U,...), split-factor overload
solveInPlace  (ref A_to_LU, ref Pivot P, ref b_to_x)       // NEW: decompInPlace + decompSolve fused driver (GESV).
                                                           // Exits are USABLE factors — keep solving via decompSolve.
```
`luDecompositionNoPivot` → `decompNoPivot` (same safe-A treatment) — see OQ-6.

### Cholesky
```
decomp        (in A, ref L)                                // rename; already safe; L-aliases-A stays documented (OQ-3)
decompSolve   (ref L, ref b_to_x)                          // = choleskySolve(ref L, ref b)
solveInPlace  (ref A_to_L, ref b_to_x)                     // NEW (POSV): aliasing decomp + decompSolve; zero scratch; exits usable
```
DELETE `choleskySolve(in A, ref L, ref b)` (2-line composition in disguise).

### CholeskyPivot (NEW CLASS — split from Cholesky)
```
decomp        (in A, ref L, ref Pivot P[, relTol])         // = choleskyDecompositionPivot; returns RankRevealingInfo
decompSolve   (ref L, in Pivot P, int rank, ref b_to_x[, relTol]) // = choleskyPivotSolve(ref L,...)
solveInPlace  (ref A_to_L, ref Pivot P, ref b_to_x[, relTol])     // replaces choleskyPivotSolve(in A,...); destructive now
```

### QR
```
decomp        (in A, ref Q, ref R[, ref u[, ref w]])       // NEW BEHAVIOR: copies A into Q (one memcpy), A preserved
decompInPlace (ref A_to_Q, ref R[, ref u[, ref w]])        // = today's qrDecomposition semantics, renamed honestly
decompSolve   (ref Q, ref R, ref b, ref x)                 // = Solvers.solveQR(Q,R,b,x); moves to QR class; b preserved, x separate (rectangular: len(b)=m != len(x)=n)
solveInPlace  (ref A, ref b, ref x[, ref u])               // = qrDirectSolve. FUSED kernel: streams Qᵀb, never forms Q.
                                                           // A, b exit as SCRATCH (R+reflectors / Qᵀb) — plain names, doc "destroyed".
```
DELETE all three `Solvers.solveQR` overloads (factor-consuming ones become `QR.decompSolve`;
the `(A,b,x)` trap alias dies).

### QRCP (NEW CLASS — split from QR)
```
decomp        (in A, ref Q, ref R, ref Pivot P[, ref u])   // NEW BEHAVIOR: A preserved
decompInPlace (ref A_to_Q, ref R, ref Pivot P[, ref u])    // = today's qrDecompositionColumnPivot semantics
solveInPlace  (ref A_to_Q, ref b, ref x, ref R, ref Pivot P, ref u[, relTol])  // replaces qrcpDirectSolve
              + allocating overload (ref A_to_Q, ref b, ref x[, relTol])
```
**OPTIMIZATION**: today's qrcpDirectSolve memcpys A into a caller-provided Q scratch.
solveInPlace factors A's buffer directly → drops the copy AND the m×n Q scratch param.
Strictly faster and leaner. A exits as the usable orthogonal factor (with R, P) → `A_to_Q`.
b preserved (read via dot). No decompSolve (rank-truncated solve is fused logic) — gap accepted.

### LQ
```
decomp        (in A, ref L, ref Q[, ref ws])               // rename only; already safe
minNormSolve  (ref A, ref b, ref x[, ref ws])              // unchanged semantics (reads A — free safety); drop lq prefix
```

### SVD / Solvers primitives
- `SVD.pinvSolve` unchanged (pinv = operation name, not redundant).
- `Solvers.solveUpperTriangular/solveLowerTriangular[LU]`: keep names (primitives beneath
  the scheme), **param `x` → `b_to_x`** (today it lies on entry).
- Iterative solvers (`cg`...): untouched.

## Contracts to document (XML, one line each)

1. Direct-solver `ref x` outputs: "output only; prior contents ignored; safe to allocate
   with `uninit: true`." (Verified true today; promote from incidental to guaranteed.)
2. Iterative-solver `ref x`: "initial guess (warm start); overwritten with solution."
   (Mostly documented already; make uniform.)
3. `solveInPlace` exits: usable-factors families (LU/Cholesky/CholeskyPivot/QRCP) document
   "A holds the factorization on return; valid input to decompSolve." QR documents destroyed.
4. `InPlace` token definition goes into docs/naming-style-guide.md.

## Deletions (recap)

- `QR.qrDirectSolve`, `QR.qrcpDirectSolve` (renamed/absorbed)
- `Solvers.solveQR` ×3
- `Cholesky.choleskySolve(in A, ref L, ref b)`
- destructive `luDecomposition(ref U, ref L, ref P)` middle form (subsumed by safe decomp + compact decompInPlace)
- `Decomposition`/`DirectSolve` name tokens from the dense API entirely

Kept as-is: `DirectSolveInfo`/`DirectSolveStatus`/`RankRevealingInfo` (direct = solver
taxonomy vs `IterativeSolveStatus`; correct usage).

## Follow-up commit: fProxyQRCache

QR is the only family without a Cache struct (15 exist). New `fProxyQRCache` carrying
`u`, `w` + the 5 blocked-WY buffers (`Vpanel`, `Tbuf`, `Wbuf`, `tcolBuf`, `VfullBuf`)
+ arena factory. Effect: zero-alloc overloads gain the level-3 blocked path (today locked
to allocating overloads by contract), and `solveInPlace` stops Temp-allocating per call —
may also claw back the measured small-N float overhead (35.4→37.2ms A/B vs eadf6a8).
Overloads become `(..., ref fProxyQRCache cache)` replacing raw `ref u, ref w`. QRCP shares
it (OQ-7).

## Test plan

- Mechanical: rename sweep across TemplateSourceTests; regen; suite must stay green (4575).
- New tests: (a) `decomp` preserves A (LU/QR/QRCP — the behavior changes); (b) `solveInPlace`
  exits are valid factors (LU/Cholesky/CholeskyPivot/QRCP: follow-up decompSolve matches
  fresh solve bit-for-bit); (c) uninit-x contract (fill x with NaN sentinel, solve, assert
  clean result — direct solvers only); (d) QRCP.solveInPlace result bit-identical to old
  qrcpDirectSolve on full-rank + rank-deficient cases.
- Benchmarks: QRVariantsBenchmark method names updated; verify QRCP no-copy shows up
  (expect small win at 2048×512: memcpy + scratch drop).

## Commit plan

1. Mechanical renames + class splits (QRCP, CholeskyPivot) + param renames + SVD/Eigen
   bare-name sweep + doc contracts. Pure-composition/trap deletions land here too
   (`Solvers.solveQR(A,b,x)` alias, `choleskySolve(in A, ref L, ref b)` — call sites
   rewritten to the explicit composition). No semantic change to surviving methods.
   Suite green. NOTE transitional state: `QRCP.solveInPlace` keeps the old (copying)
   signature in this commit; its name matches semantics only after commit 2.
   Destructive `luDecomposition(ref U, ref L, ref P)` transitionally becomes a
   `decompInPlace(ref A_to_U, ref L, ref P)` overload (arity-distinct from compact form);
   deleted in commit 2.
2. Behavior changes: safe `decomp` variants (LU/QR/QRCP) + remaining deletions + new
   `solveInPlace` drivers (LU/Cholesky/CholeskyPivot) + explicit `Cholesky.decompInPlace`
   + QRCP no-copy optimization. New tests. Also delete the [Obsolete] Jacobi
   `svdDecomposition` (pre-release, no shims policy).
3. `fProxyQRCache` + level-3 on zero-alloc path. Benchmark re-run.
4. Docs sweep: README, docs/features/*, naming-style-guide.md.

## Open questions — ALL RESOLVED 2026-07-05

- **OQ-1 (scope)**: ✅ YES — SVD/Eigen ride along in commit 1. `SVD.svdThin`→`thin`,
  `svdValues`→`values`, `svdTruncated`→`truncated`, `svdRandomized`→`randomized`
  (`pinvSolve`, `pinv`, `lowRankApprox` unchanged — operation names, not class echoes);
  `Eigen.eigenSymmetric`→`symmetric`, `eigenvaluesSymmetric`→`valuesSymmetric`, and any
  other `eigen*`-prefixed members. `powerIteration`/`lanczos`/etc. already bare — keep.
- **OQ-2 (FFT/other classes)**: ✅ untouched. `FFT.fft`, `LOBPCG.lobpcg` stay — the class
  IS the operation name, so the echo is the operation, not redundancy.
- **OQ-3 (Cholesky in-place)**: ✅ add explicit `decompInPlace(ref A_to_L)`. The
  L-aliases-A trick stops being the documented path (hidden aliasing contracts are what
  this rework kills).
- **OQ-4 (pivot params)**: ✅ caller-provided `ref P` only. `out P` REJECTED: out forces
  the method to choose the allocator, transfers Dispose ownership implicitly (leak bait),
  and breaks the pattern that caller-sized buffers are validated size contracts.
- **OQ-5 (QR safe decomp cost)**: ✅ proceed — copy target Q must exist anyway; cost is
  one memcpy.
- **OQ-6 (luDecompNoPivot)**: ✅ keep; rename treatment; docs warn.
- **OQ-7 (QRCP cache)**: ✅ decide during commit 3, whichever avoids dead fields.
- **OQ-8 (lqMinNormSolve destructive variant)**: ✅ deferred.
- **OQ-9 (migration)**: ✅ hard rename, no [Obsolete] shims.
- **OQ-10 (multi-RHS)**: ✅ deferred to a future spec, after this rework lands.
