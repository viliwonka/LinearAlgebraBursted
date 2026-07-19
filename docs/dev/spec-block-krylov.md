# Mini-spec: TRUE block-Krylov (multi-RHS) solvers

## 0. Scope and source-of-truth

Templates live under `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/`; codegen reads only `*.cs`, so all
new code goes in a template `.cs`, comments state contracts only, rationale goes in
`TemplateSource/OP/DEVLOG.md`. This is the block follow-up the DEVLOG flagged: task #12 / the design
directive ("~7 UNIFIED block algorithms, identity default, no p-prefix, on ApplyBlock +
QRCP-orthogonalized blocks (tall-skinny, O(n·s²)) + s×s coeff via CHOP/QRCP; block-preconditioner =
column-loop helper; SKIP cgls/cgne normal-equations methods").

Goal: a TRUE block recurrence (one shared Krylov subspace built from all s RHS at once, block
coefficients), NOT s independent scalar solves. Delivers the block-Krylov convergence advantage (all RHS
converge in ≤ the worst single-RHS iteration count) and streams A once per iteration via `ApplyBlock`.

New file: `OP/Krylov.Block.fProxy.cs` (dense + BSR + generic overloads for block-CG first). Partial-class
`Krylov`.

## 1. Block data model (B and X as `fProxyMxN`, ROW-major, s rows = s RHS)

A "block vector" is an `fProxyMxN` with **s rows × n cols**: row j is RHS/solution vector j (`X[j,:]` is
the j-th length-n vector). Exactly how LOBPCG holds X/W/P blocks and what `ApplyBlock` consumes.

- `ApplyBlock(in fProxyMxN Vrows, ref fProxyMxN AVrows, int rows)` — `LinearOperator.fProxy.cs:44`.
  `AVrows[i,:] = A·Vrows[i,:]` for the first `rows` rows; AVrows must not alias Vrows; symmetric-A fast
  path. Dense = one `Blas.dotRows` GEMM (`LinearOperator.fProxy.cs:100`). The recurrence applies A to all s
  columns per iteration via a single `ApplyBlock`, never s separate `Apply` calls.
- Block inner products (s×s Gram Vᵀ·W): with row-major s×n blocks, `G[i,j]=dot(V[i,:],W[j,:])` = V·Wᵀ.
  Reuse LOBPCG `FillGramSub` (`LOBPCG.fProxy.cs:1125`, symmetric same-block fast path) or `Blas.dotSymT`
  (`OP.Dot.fProxy.cs:302`); general s×s = `Blas.dot(V,W,transposeA:false,transposeB:true)`
  (`OP.Dot.fProxy.cs:208`).
- Block update X += P·C (C is s×s): combine s row-vectors by an s×s coeff. Shape of LOBPCG
  `SvqbAccumulate` (`LOBPCG.fProxy.cs:1103`) / a small `matMatDot` (`UnsafeOP.fProxy.cs:485`).
- **Convention decision:** public B/X block is **s-rows × n-cols (row = RHS)**, matching `ApplyBlock`.
  The column-major convention (`CHO.decompSolve` stores RHS as columns, `CHO.fProxy.cs:274`) appears ONLY
  internally at the tiny s×s coefficient solve (§3), where the coder transposes explicitly. Validate
  `B.M_Rows == X.M_Rows == s`, `B.N_Cols == X.N_Cols == A.Rows`.

## 2. Block orthogonalization + deflation (THE critical correctness point)

Block-CG (O'Leary 1980) breaks down when the s residual columns become linearly dependent — the s×s
`PᵀAP` goes singular. Proper fix = **breakdown-free / deflated block-CG** (Dubrulle 2001; Ji & Li 2017):
each iteration rank-reveal the residual/search block and **drop dependent columns**, carrying an **active
width `sa ≤ s`** exactly like LOBPCG locking (`numActive`).

Reuse, do not reinvent:
- Tall-skinny orthonormalization of n×s, O(n·s²), with dropping — LOBPCG `OrthonormalizeBlockB`
  (`LOBPCG.fProxy.cs:1029`, SVQB, B-inner-product) and `OrthonormalizeBlock` (`:982`, Euclidean, M=I).
  Both return surviving-row count `kept`, writing kept rows into the LEADING rows, leaving `[kept,s)` stale
  — that IS the deflation mechanism. SVQB: Gram scaled by diag⁻¹ᐟ², symmetric eig via
  `Eigen.symmetricInPlace`, keep θ_j > θ_max·s·ε·10 — drops rather than ridge-inflates.
- DEVLOG directive alternative = QRCP (`QRCP.decomp` `:708`, `RankInfo.rank` = deflation count).
  **Recommendation: reuse `OrthonormalizeBlock`/`OrthonormalizeBlockB`** for block-CG (return kept-count,
  carry A-image/B-image blocks by the same combination for free, proven in a Burst job). Reserve QRCP for
  nonsymmetric block solvers.

Deflation policy (concrete):
1. Form residual block R (n×s), Z=M⁻¹R, orthonormalize the search block; primitive returns `sa`.
2. Track `sa` = active width; subsequent P, AP, Q use `sa` rows. Dropped columns' partial solutions in X
   are already committed and keep converging through the shared subspace — do NOT zero them.
3. `sa == 0` → block subspace exhausted → `MaxIterations`/`Breakdown` for not-yet-converged columns.
4. Per-column convergence tested independently. **First cut: keep-all-columns, per-column convergence
   flags, stop when all flagged.** Locking (Swap.Rows to back, shrink active width, `Deflate`
   `LOBPCG.fProxy.cs:1173`) is a later optimization; note in DEVLOG.

## 3. The s×s coefficient solves (α, β are s×s matrices)

- **Block-CG (SPD):** coeff matrix `PᵀAP` (sa×sa, SPD) → **`CHO.decomp`/`CHO.decompSolve`**
  (`CHO.fProxy.cs:27,:274`). If P orthonormalized first (recommended), PᵀP=I, systems near-identity/
  well-conditioned. If factoring raw PᵀAP, use `FactorGram` ridge-retry discipline (`LOBPCG.fProxy.cs:1261`).
- **Nonsymmetric (block-BiCGStab / block-GMRES):** general sa×sa → **QRCP**
  (`QRCP.solveInPlace` multi-RHS `:1431`) or LU; `CHOP` (`CHOP.decompSolve` `:650`) where SPD-but-semidef.
- **Determinism:** s×s solves are fixed-order scalar/`matMatDot` under Strict — deterministic. Tall-skinny
  reordered dots fall under the pre-1.0 deterministic-reorder waiver (as LOBPCG).
- **Convention gotcha:** CHO/CHOP/QRCP multi-RHS treat each RHS as a **column** of n×s `B_to_X`
  (`CHO.fProxy.cs:273`). Block vectors are row-major (s×n). At the s×s solve, build coeff matrix + RHS in
  column-RHS orientation, solve, apply s×s coeff back to row-major blocks. s×s (s≤~32) so explicit
  transpose is negligible — spell it out to avoid a silent transpose bug.

## 4. Block preconditioner

Apply M to an n×s block. **Column-loop over `M.Apply` is acceptable** (preconditioner, not the recurrence
— the A-applies still go through the single `ApplyBlock`). Precedent: LOBPCG per-row
`M.Apply(in ws.rowIn, ref ws.rowOut)` (`LOBPCG.fProxy.cs:407-412`). Write one helper
`ApplyBlockPre<TPre>(in TPre M, in fProxyMxN Rrows, ref fProxyMxN Zrows, int rows)` looping rows through
`M.Apply` into two Temp row scratch. No per-preconditioner rewrite.

**Identity folds out** like the scalar family: `IsIdentity` (`LinearOperator.fProxy.cs:64`) compile-time
literal; under `fProxyIdentityPreconditioner` (`:113`) the `if(!M.IsIdentity) ApplyBlockPre(...)` branch
folds and Z aliases R (no copy, no block scratch). Same mechanism as merged `cg<TOp,TPre>`
(`Krylov.fProxy.cs:128`, body `:190-249`). A dedicated block ApplyBlock-on-preconditioner is not worth it
first cut — note as later optimization.

## 5. Algorithms, in order

1. **block-CG (SPD) — FIRST.** Establishes every block primitive. Parallels `cg<TOp,TPre>`
   (`Krylov.fProxy.cs:128`). Block diff: α/β are sa×s; breakdown = PᵀAP rank loss → residual-block
   deflation (§2), not scalar `pAp ≤ 0`.
2. **block-MINRES (symmetric indefinite).** Parallels `minres<TOp,TPre>` (`:526`). Block Lanczos + banded
   QR; deflation on the s×s Lanczos off-diagonal. Higher effort (banded Givens bookkeeping).
3. **block-BiCGStab (nonsymmetric).** Parallels `biCGStab<TOp,TPre>` (`:811`). Block shadow residual;
   block ρ / rHat0ᵀV singular / block ω via QRCP.
4. **block-GMRES(m).** Parallels `gmres<TOp,TPre>`. Block Arnoldi (orthogonalize s new vectors, deflate),
   block Hessenberg least-squares. Own Temp workspace.

Deliver block-CG complete (green + tested + benchmarked) before the next. Do NOT build
block-cgls/block-cgne (normal-equations, κ²).

## 6. Single-body + IsIdentity per algorithm

- Core `cg<TOp,TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X, <block scratch>, int maxIter,
  fProxy tol)`, constraints `TOp:struct,IfProxyLinearOperator`, `TPre:struct,IfProxyPreconditioner`.
- Unpreconditioned forwarder `cg<TOp>(... no M ...)` → core with `default(fProxyIdentityPreconditioner)` +
  `default` Z block (never dereferenced under fold), like `cg<TOp>` `Krylov.fProxy.cs:34-43`.
- Concrete dense/BSR forwarders wrap `fProxyDenseOperator`/`fProxyBSROperator` + arena-allocating +
  default-param rungs (3-rung ladder).
- No `pblockCg`/`blockPcg`. Every M-block access behind `if(!M.IsIdentity)`. Add
  `BlockCgIdentityMatchesPlain` bit-identity test (mirror `MergedCgIdentityMatchesPlainCg`).

## 7. Naming / signatures / diagnostics

- **Name: overload `cg`** on `in fProxyMxN B, ref fProxyMxN X` (library overloads by arg type; the direct
  multi-RHS solvers already do this — `CHO.decompSolve`/`QRCP.solveInPlace` fProxyN vs fProxyMxN). Short
  param names `maxIter`,`tol`.
- **Diagnostics: dedicated `BlockSolveInfo`** (non-templated `OP/BlockSolveInfo.cs` — type-agnostic, avoids
  CS0102). Fields: `int iterations`, `int converged`, `int rhs`, `double maxRnorm`,
  `IterativeSolveStatus status`, `Solved => status==Converged`, implicit bool, `ToFixedString()`. Model on
  `SolveInfo` (`SolveInfo.cs:79`) + `LOBPCGInfo`. Reuse `IterativeSolveStatus` (`SolveStatus.cs:16`).
- **Scratch:** caller-owned cache struct allocated once (`fProxyBlockKrylovCache`, analogous to
  `fProxyLOBPCGCache`) holding R/P/AP=Q/Z (O(n·s)); allocating convenience overload via `fProxyTempMat`.
  s×s coeff matrices are `Allocator.Temp`.

## 8. Acceptance criteria

1. **vs scalar:** block-CG on s RHS matches s scalar `cg` per-column to tolerance (per-column residual ≤
   tol·‖b_j‖, NOT bit-equality). Oracle: loop `Krylov.cg(in A, B[j,:], ...)`.
2. **Block advantage:** on a well-separated SPD system, `blockInfo.iterations <= max over columns of scalar
   iters`. Gallery matrix with clustered-but-separated spectrum.
3. **Identity no-op:** `cg<…,fProxyIdentityPreconditioner>` Burst-compiles and is bit-identical to the
   plain block body (exact double-equality on X/iterations/status, fixed seed).
4. **Deflation:** rank-deficient RHS block (two identical columns) must NOT NaN/throw — deflates (`sa`
   drops), solves every consistent column, honest `status`, finite X.
5. **Job-safe:** run through `IJob.Run()` (by-value copy), caller sees final X (the RestoreBufferIdentity /
   cache-reseat hazard LOBPCG hit, `LOBPCG.fProxy.cs:117-124`).
6. Suite green; regen compiles float+double clean; add `BlockKrylovBenchmark` (block-CG vs s scalar, N and
   s swept) under `TemplateSourceBenchmarks`, `Tools/benchmark.ps1`.

### Reference algorithms
O'Leary 1980 (block CG recurrence); Dubrulle 2001 / Ji & Li 2017 (breakdown-free deflated variant §2's
drop-columns policy implements).
