# Audit: LQRP pivoted-output footgun (task #51) + 0*NaN uninitialized-buffer sweep (task #52)

Read-only investigation. No code changed. Both audits stem from bugs found and fixed in the
(currently uncommitted) `bminres` block-MINRES solver:
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.MINRES.fProxy.cs`.

---

## Audit 1 -- LQRP row-pivoted output: caller correctness

### The documented contract

`LQRP.decomp` factors `P.A = L.Q` with a ROW permutation `P` (transpose-dual of `QRCP`'s column
pivoting). The contract IS stated precisely on `decomp`'s own XML doc, not left implicit:

- `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LQRP.fProxy.cs:248-249` -- `P`'s doc: `(P.A)[j, :]
  == A[P[j], :], equivalently A[P[j], :] == (L.Q)[j, :]`.
- Class remarks at `LQRP.fProxy.cs:11-19` state the same equation and rank-revealing rationale.
- `solveInPlace`'s own body is the canonical "how to consume a row permutation" reference:
  `LQRP.fProxy.cs:507-508` gathers `v[j] = b[P[j]]` (RHS into pivoted order) before the triangular
  solve, and its doc (`LQRP.fProxy.cs:522-524`) notes columns are untouched so `x` needs no
  un-permute -- only `b` (equivalently, anything indexed by the ROW dimension) does.

What is not spelled out as its own sentence: that `Q`'s ROWS are also in pivoted order (row
`j` of `Q` is the `j`-th pivot step's reflector output, not row `j` of the un-permuted input) --
this is only derivable by combining the `P` doc with the `P.A = L.Q` equation, not stated
independently next to the `Q` param doc (`LQRP.fProxy.cs:247`: "Output m x n row-orthonormal
factor", no mention of ordering).

`LQRPTests.fProxy.cs` (`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LQRPTests.fProxy.cs:381-392`)
already asserts the documented reconstruction (`Aperm[j,c] = A[P[j],c]` then compares against
`L*Q`) and a dedicated `FirstPivotLargestRow` test checks `P[0]` picks the original row of largest
norm -- the contract is both stated and tested. The bminres bug was a misuse of `L` (consumed
as a recurrence coefficient matrix indexed by original-RHS order without applying `P`), not a
misreading of an undocumented contract.

### Caller-by-caller table

All functional (non-test, non-doc-comment) callers of `LQRP.decomp` / `decompInPlace` /
`solveInPlace` / `minNormSolveInPlace` / `decompSolve` / `minNormDecompSolve` outside
`LQRP.fProxy.cs` itself, found via exhaustive grep of `Assets/LinearAlgebra` for `LQRP\.`:

| # | Caller | File:line | Consumes row order? | Correct? |
|---|---|---|---|---|
| 1 | `BlockNormalizeIdentity` (bminres unpreconditioned Lanczos normalize) | `Krylov.Block.MINRES.fProxy.cs:65-75` | YES -- `Beta` (=`L`) is used as a recurrence coefficient matrix matched to `Vprev`/`Alfa`/`OmegaOld`'s original-RHS row order | Fixed. `UnpivotBetaRows` (`Krylov.Block.MINRES.fProxy.cs:45-56`) scatters `Beta`'s rows back through `P` before any consumer reads it. `Vout` (=`Q`) is used only as an arbitrary orthonormal basis (its row index has no semantic meaning outside this call), so it correctly needs no un-permute -- only `Beta` did. |
| 2 | `BlockNormalizePrecond` (bminres preconditioned normalize) | `Krylov.Block.MINRES.fProxy.cs:86-116` | Same reasoning, via `CHOP.decomp` (pivoted Cholesky) not `LQRP`, but identical row-permutation shape | Fixed -- same `UnpivotBetaRows` call at line 113. Listed for completeness since it shares the bug class and the fix. |
| 3 | `bgmres` restart-cycle basis factor | `Krylov.Block.GMRES.fProxy.cs:156-160` (`LQRP.decomp(in R0, ref L0, ref Q0, ref Ppiv0)`) | NO -- `L0`'s only consumer is `LQRPRank(in L0, ...)` (`Krylov.Block.Common.fProxy.cs:167-178`), which reads only `abs(L[i,i])` down the diagonal -- a permutation-invariant property (pivoting only changes which row holds the i-th diagonal magnitude, not the non-increasing magnitude sequence itself). `Q0` is used purely as an arbitrary orthonormal basis for the residual's row space (`V0 = RowsView(Q0, w[0])`), never matched back to `R0`'s original rows. | Safe. `Ppiv0` is disposed immediately, unused -- correctly so, since nothing downstream needs it. |
| 4 | `bgmres` per-Arnoldi-step deflating LQ | `Krylov.Block.GMRES.fProxy.cs:205-211` | Same as #3 (`Lv` used only for `LQRPRank`; `Qout` used only as arbitrary basis for the next Arnoldi vector block) | Safe, same reasoning. |
| 5 | `FactorLiveResidual` (bcgrq's live-residual orthonormalize) | `Krylov.Block.Common.fProxy.cs:185-209` | Same shape: `Lv` consumed only via `LQRPRank`; `Qfull` (written into `Pa`) consumed only as an arbitrary orthonormal search basis in `bcgrq` (`Krylov.Block.BCGrQ.fProxy.cs:90-104`, `138-165`) via fresh GEMMs (`Blas.dot(in Psearch, in Rlive, ...)` etc.) -- never assumes `Pa`'s row i corresponds to `R`'s original row i | Safe. `Ppiv` disposed unused, same as #3/#4. |
| 6 | `FactorLiveSearch` (bfbcg's live-search orthonormalize) | `Krylov.Block.Common.fProxy.cs:214-225`, called from `Krylov.Block.BFBCG.fProxy.cs:101,171` | Same shape as #5 | Safe, same reasoning. |
| 7 | `TallWideSolveBenchmark` (`WideLQRPJobFProxy`/`WideLQRPSolveJobFProxy`/`WideLQRPMinNormJobFProxy`) | `Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/TallWideSolveBenchmark.fProxy.cs:99-147` | N/A -- calls the solve-level API (`LQRP.solveInPlace`/`minNormSolveInPlace`/`decomp`) directly, which internally handles the permutation per its own contract; the benchmark never reconstructs anything from raw `L`/`Q`/`P` itself | Safe by construction -- nothing to get wrong here. |

No other functional callers exist (confirmed by grepping all of `Assets/LinearAlgebra` for
`LQRP\.`, 30 files total: the 7 above, `LQRP.fProxy.cs`/`LQRP.Workspace.fProxy.cs` themselves,
their generated `Source/OP/*.{float,double}.cs` mirrors, `QRCP.fProxy.cs`/`LQ.fProxy.cs` (comment
mentions only, no calls), and the test files `LQRPTests.fProxy.cs`/`MatrixViewInvariantTests.fProxy.cs`).

### Why the bug happened despite a documented contract

The pattern that failed (#1/#2) is structurally different from the ones that were always safe
(#3-#6): it is the only call site that reuses the pivoted factorization's `L`/`Beta` output as
a numeric coefficient matrix tied to a specific, externally-meaningful row index (the block-RHS
index shared with `Vprev`/`Alfa`/`OmegaOld`). Every other caller uses `LQRP` purely as a
"rank-revealing orthonormalization black box": it wants a fresh orthonormal basis (`Q`, row order
irrelevant) and a rank/magnitude signal (`L`'s diagonal, permutation-invariant) -- never `L`'s
off-diagonal values indexed against anything outside the call. This is easy to get right by
accident (nothing to permute) and easy to get wrong on purpose (permutation matters, and it is not
flagged at the point of use).

### Recommendation: (a) documentation-only clarification, plus reusing the existing fix as the canonical pattern

The contract is already correct and already tested; the four safe callers show LQRP's normal
consumption pattern doesn't even touch the permutation. A new "unpivoted convenience overload"
(option b) would be speculative API surface for a need that has arisen exactly once, in exactly
one place, and was fixed locally -- against the project's stated preference for small completable
units over anticipatory API growth. Recommend:

1. Strengthen `LQRP.decomp`'s doc (`LQRP.fProxy.cs:237-249`, mirrored on `decompInPlace` and
   the workspace overloads) with one explicit sentence on the `Q` param and/or class remarks:
   "L's and Q's ROWS are BOTH in pivoted order (row j = the j-th pivot step), not input order. A
   caller that reuses L as a coefficient matrix tied to an externally-meaningful row index (rather
   than treating Q as an arbitrary orthonormal basis) must gather/scatter through P -- see
   `solveInPlace`'s `v[j] = b[P[j]]` gather, or `Krylov.Block.MINRES.fProxy.cs`'s
   `UnpivotBetaRows` for the L-scatter case." This puts the exact failure mode (not just the
   permutation equation) where every future reader of the contract will see it.
2. Optionally promote `UnpivotBetaRows` (currently a private static helper local to
   `Krylov.Block.MINRES.fProxy.cs:45-56`) to a shared location (e.g.
   `Krylov.Block.Common.fProxy.cs`, alongside `LQRPRank`) once a second consumer needs it -- not
   urgent today (only one caller), but cheap to relocate later and saves the next author from
   re-deriving the same scatter. Not required to close this audit; flagged as a low-priority
   follow-up, not a blocking recommendation.
3. No code changes to `LQRP.fProxy.cs` itself or any of the six verified-safe callers.

---

## Audit 2 -- 0*NaN / uninitialized-buffer sweep (Krylov / Eigen / preconditioners)

### Methodology

`fProxyMxN`/`fProxyN`'s allocating constructors take a `bool uninit = false` 4th argument
(`Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/fProxyMxN.cs:117-128`): `false` ->
`NativeArrayOptions.ClearMemory` (zeroed), `true` -> `NativeArrayOptions.UninitializedMemory`
(garbage -- real prior heap/stack content, not necessarily "random," which is exactly how the
bminres bug reproduced: a leftover NaN bit pattern from an earlier allocation, not synthetic
noise). Swept every `Allocator.Temp/Persistent, true)` call (186 occurrences across 17 `OP/*.cs`
files) plus the block-Krylov/LOBPCG/preconditioner-setup files, and for each judged whether the
first use fully covers the buffer before any read, or whether a read/multiply/accumulate can hit a
region that was never (fully) written first -- the exact bminres shape: an explicit "this
coefficient is zero at the boundary" branch immediately followed by an operation that still
touches a paired buffer arithmetically (`0 * garbage`) instead of skipping the touch entirely.
Per the task brief, only genuinely suspicious sites are itemized below; the (large) remainder that
is trivially safe (immediate full GEMM/copy overwrite before any accumulate) is summarized, not
enumerated line-by-line.

### Ranked findings

| Rank | Site | Buffer(s) | Verdict | Reasoning |
|---|---|---|---|---|
| 1 (confirmed, fixed) | `bminres` k=0 search-direction recurrence | `W`/`W1`/`W2` (sxn, `Krylov.Block.MINRES.fProxy.cs:237-244` allocated `true`) | Was a real bug; now fixed. | `Krylov.Block.MINRES.fProxy.cs:321-322` explicitly zeroes `W1`/`W2`/`W` before the loop, with a comment naming the exact mechanism ("0 * NaN is NaN, not 0"). At k=0, `Delta`/`OldEps` are algebraically zero (lines 312-313), so `BlockCTV(in Delta, in W2, ...)` etc. should contribute nothing -- but only does so if `W2` itself holds `0.0`, not garbage. This is the reference case the rest of this sweep is checking against. |
| 2 (historical, closed, same root-cause family) | `bgmres`/`LQRP.decomp`/`QR.decompInPlace` zero-threshold | `Wbuf`, `HQscratch` narrowed via `RowsView`/`RectView`, fed to `Norms.LInf` | Was a real bug; already root-fixed, not by this task. | `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/DEVLOG.md:147-168` (Krylov.bgmres) + `:136-145` (NormsOP): `Norms.LInf` used to scan a view's true allocated `Data.Length` instead of its logical `M_Rows*N_Cols`, over-reading into an uninitialized/stale tail and corrupting the Householder zero-threshold. Root-caused (`:122-134`, Krylov.Block.Common / LOBPCG reslice fix) via a real-view constructor + `Norms.*` now scanning logical extent -- structurally closed, not just patched at the two original call sites. Listed because it is the other known instance of "Temp scratch feeds a numeric threshold without full-coverage guarantee," and the DEVLOG explicitly warns future narrowed-view buffers need the same care (`:163-168`). |
| 3 (informational, different class -- not uninit/NaN) | LOBPCG `UpdateActiveBlock` | `ws.Pnext` locked-row tail, rows `[numActive, k)` | Not a bug. | `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LOBPCG.fProxy.cs:1465-1503`: `ws.Xnext` is fully written for ALL k rows (active combine, lines 1465-1482, plus explicit carry-forward of locked rows, lines 1498-1500) before `SwapMat(ref ws.X, ref ws.Xnext)`. `ws.Pnext`, however, is written only for the active rows `[0, numActive)` (lines 1484-1495) before `SwapMat(ref ws.P, ref ws.Pnext)` -- its locked-row tail keeps whatever `Pnext` held from an earlier ping-pong cycle. This is stale-but-finite data from a persistent cache buffer (never `uninit:true`-allocated per call), not garbage -- and `numActive` only shrinks, so those stale rows of `P` are structurally never read again (LOBPCG's active-block Gram/update always narrows to the current, smaller `numActive`). Flagged only because it is the closest structural cousin to the bminres shape (swap-based ping-pong buffer with a partial write) found anywhere else in the sweep -- worth a second look if `UpdateActiveBlock`'s row-range logic ever changes. |
| -- (checked, safe) | `bgmres`/`bcgrq`/`bfbcg`/`bbiCGStab`/`bcg` sxs and sxn coefficient scratch | `Alfa`/`Beta`/`Dbar`/... (bminres, all fixed by #1's zeroing), `Mmat`/`Rhs`/`alphaMat`/`betaMat`/`PQ`/`RZ`/... (the other four block solvers) | Safe. | Every `Allocator.Temp, true)` scratch in these five files is fully overwritten by a `BlockGram`/`BlockCTV`/`CopyBlock`/`CopyMat` GEMM-or-copy call before its first read; none is touched via `BlockAdd` (`+=`) as its first operation. `bgmres`'s `Hbuf`/`Gbuf` are the only ones that look risky at a glance (accumulated into via `StoreBlockAt`, `+=`) but are explicitly `ZeroPrefix`-cleared over their entire extent every restart cycle first (`Krylov.Block.GMRES.fProxy.cs:153-154`), so unwritten regions are genuine zero, not garbage -- this is the write-fully-then-accumulate pattern done correctly. |
| -- (checked, safe) | `Krylov.MINRESQLP.fProxy.cs` 3-term search-direction history | `w`/`wl`/`wl2`/`xl2` | Safe, proactively guarded. | `Krylov.MINRESQLP.fProxy.cs:128` zeroes all four explicitly before the loop -- the same fix shape as #1, applied up front rather than discovered by a failing test. `t1` needs no zeroing: every use writes it fully via `Blas.scaledCopy` (full overwrite) before any `addScaledInPlace` accumulate. |
| -- (checked, safe) | Preconditioner setup: `fProxyFSAI`/`fProxySPAI`/`fProxyBlockJacobi`/`fProxySchwarz` | `Ahat`, `M` (nxn, `Allocator.Temp, true`, only lower triangle filled) | Safe by an existing library-wide contract. | `CHO.decomp`/`decompInPlace` (`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/CHO.fProxy.cs:54-58`) unconditionally copies-or-zeroes both triangles as its very first act -- `L[i,j] = A[i,j]` for `j<=i`, `L[i,j] = 0` for `j>i` -- before any positive-definiteness check can fail out. So any caller that fills only a Cholesky input's lower triangle and leaves the upper triangle `uninit:true` is safe by construction, and (in `fProxySchwarz.cs:328-329`) copying the entire post-factorization buffer (including the now-zeroed-not-garbage upper triangle) into a packed `factors` array is also safe. `fProxySPAI.cs`'s `Ahat`/`Bt`/`Bwork`/`Cwork` are all fully (not triangularly) written by explicit double loops before `QR.solveInPlace`. `fProxyBlockJacobi.cs`'s `Dcopy`/`col` are fully written before `LU.decompInPlace`/`decompSolve`. |
| -- (checked, safe) | `Krylov.IDR.fProxy.cs` | `Msys` (allocated `Allocator.Temp, false` -- deliberately cleared, not uninit) | Safe, and instructive. | `Krylov.IDR.fProxy.cs:79-80` explicitly chose `uninit:false` here (unlike the sxs buffers in the other block solvers) with the comment "rows above k stay untouched -- always zero, never read" (`:154`) -- because `Msys`'s lower-triangular fill is incremental across iterations (not a single self-zeroing factorization call like CHO), so it cannot rely on a callee's zeroing contract and correctly clears at allocation instead. A good example of the right call being made deliberately. |
| -- (checked, safe) | LOBPCG active/search-direction reads | `ws.P`/`ws.AP`/`ws.BP` | Safe, stronger idiom than bminres's original. | Gated behind an explicit `if (useP)` (`LOBPCG.fProxy.cs:1225`, `1301-1310`, `1477`, `1490`) rather than relying on a zero coefficient to arithmetically cancel a read -- the read itself is skipped when the block doesn't yet apply, which cannot be defeated by NaN/Inf garbage the way a "multiply by zero" guard can. |

### Recommendation

No further code fix is warranted from this sweep -- the one confirmed live instance (bminres) is
already fixed, and no second live instance was found in Krylov/Eigen/preconditioner code despite a
targeted, adversarial search for the same shape (boundary-zero coefficient + paired buffer touch,
and swap/ping-pong buffers with partial writes). Two low-cost preventive suggestions for the
owner to weigh, neither blocking:

1. Write a short DEVLOG/style-guide note (mirroring the existing NormsOP entry's own
   "any future buffer that gets narrowed... needs the same treatment" warning) capturing this
   exact rule for future block/recurrence solvers: "a buffer written only conditionally (e.g. via
   a boundary `if` or narrowed to fewer than its allocated rows) must either be allocated
   `uninit:false`, be explicitly zeroed over its full extent before the loop, or have every read
   site gated behind the same condition that skipped the write (LOBPCG's `useP` pattern) -- never
   rely on `0 * x == 0` to make an unwritten touch harmless." This is process debt, not code debt:
   the rule is already followed correctly in five of the eight files checked; it just isn't
   written down anywhere a new solver's author would find it before making the same mistake a
   third time.
2. Optional, not recommended without owner sign-off: an allocation-time NaN-poison for
   `uninit:true` requests under a debug-only define, so a future instance of this bug fails loudly
   in tests instead of silently producing a wrong-but-finite answer. Distinct from the
   dispose-time NaN-poison that was removed as dead code on 2026-07-05
   (`Assets/LinearAlgebra/CodeGen/TemplateSource/fProxy/DEVLOG.md:86-94` -- that one guarded
   use-after-free, superseded by the record table's own generational/double-free checks; this
   would guard use-before-full-write, a different failure mode). Flagged as an idea only, given
   the project's stated preference (agent-cost-economy, library-todo memory notes) against
   speculative infrastructure without a repeat offender to justify it.