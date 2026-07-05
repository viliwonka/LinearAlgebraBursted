# Phase-2: solver/decomposition workspace overloads — spec

*Historical document — method names predate the 2026-07 solver-API rework (see
docs/spec-solver-api-rework.md for the mapping).*

Goal: every solver/decomposition that allocates an internal scratch buffer gains a
caller-provided-scratch overload so a hot loop (solve many same-sized systems) can hoist
the workspace allocation out of the loop and run zero-alloc. Mirrors the phase-1 ref-dest
op work (`docs/zero-alloc-ops.md`) and the Conjugate Gradient primitive+convenience split.

## Decisions (locked)

- **Primitive + wrapper (hard rule):** the scratch-taking form is the implementation; the
  existing allocating form becomes a thin wrapper that allocates the scratch (same allocator
  it used before — `Allocator.Temp` for `new fProxyN(...)` workspaces, arena `tempfProxyVec`
  for arena temps) and delegates. No duplicated math.
- **Naming:** overload the existing method with trailing `ref <scratch>` parameter(s)
  (e.g. `qrDecomposition(ref Q, ref R, ref u)`). No new suffix.
- **Exact scratch sizing.** Internal kernels use the scratch vector's `.N` as an active loop
  bound tied to a matrix dimension, so the scratch must be sized EXACTLY (not `>=`). The
  primitive throws `System.Exception`/`ArgumentException` (matching the file's existing guard
  style) if a scratch buffer is mis-sized. Hot-loop callers solving same-sized systems (the
  common case) allocate one exact-size workspace and reuse it.
- **Templates:** write overloads in `*.fProxy.cs` so codegen emits float/double automatically.
- **Tests:** for each scratch overload, a test asserting the scratch result equals the
  allocating result on random input (Burst IJob, per-precision `Tol`), plus a managed
  guard test that a mis-sized scratch throws.
- **Green per chunk:** one solver family at a time → `regen.ps1` + `run-tests.ps1`, never red.

## Inventory (check off as done)

- [x] `OP/Ortho_OP.fProxy.cs` — `qrDecomposition(ref Q, ref R, ref u)`, `qrDirectSolve(ref A, ref b, ref x, ref u)`
      (u length == M_Rows). Existing 2-arg / 3-arg forms now allocating wrappers. Tests green
      (`OrthoWorkspaceTests.fProxy` — equiv + mis-sized guard, ×2 float/double; Ortho_OP suite 60/60).
      bug-hunter audit: CLEAN (math byte-identical to original; exact-size guard confirmed load-bearing —
      oversized u would OOB-read Q/A; wrappers preserve allocator/dispose/inlining).
- [x] `OP/Solvers.fProxy.cs` — `solveQR` internal `y = dot(b, Q)` temp eliminated. Old dead
      `out`-overload (0 callers) → zero-alloc ref-dest primitive `solveQR(ref Q, ref R, ref b, ref x)`
      (computes `dot(in b, in Q, ref x)` then in-place `solveUpperTriangular`; guard `x.N != Q.N_Cols`)
      + returning convenience `fProxyN solveQR(ref Q, ref R, ref b)`. (ref/out can't coexist — CS0663 —
      so the convenience returns instead of using `out`.) Tests `SolveQRSolve` (square + tall/
      overdetermined) + alias-b/bad-size guard throws; suite 72/72. bug-hunter: production code CLEAN
      (dot(in b,in Q,ref x) == old dot(b,Q); in-place back-sub correct; guards complete). Flagged test
      gaps (tall case, alias-b guard, bad-size guard) — all CLOSED.
- [x] `OP/SVD.Solvers.fProxy.cs` — `pinvSolve` / `pseudoInverse` scratch overloads. KEY: with
      k=min(m,n), `S` is len k and `M` is k×k in BOTH branches (M = V when tall, W when wide);
      only `At` (n×m) is wide-specific (pass `default(fProxyMxN)` for m>=n). Wide branch fills At via
      phase-1 ref-dest `Linear_OP.trans(in A, ref At)`. Inner loop var `k`→`kk` (method-scope k now).
      Wrappers temp-alloc S(k), M(k×k), At(n×m only if m<n) + delegate. Tests `SVDWorkspaceTests.fProxy`
      (pinv/pseudo equiv tall+wide ×2, + 4 mis-sized guards). SVD suite 111/111. bug-hunter audit:
      CLEAN on all 7 concerns (k-unification, kk-rename, M-as-V/W, trans, default-At, guards, codegen).

## Ergonomics polish (post-completion)

- [x] `OP/SVD.Workspace.fProxy.cs` — `fProxySvd_WS { S; M; At; }` struct + `Arena.fProxySvd_WS(m, n)`
      factory (sizes S=k, M=k×k, At=n×m only if wide; k=min(m,n)). Removes the caller-side
      footgun of computing k / sizing the three buffers / remembering `default`-At for tall.
- [x] `OP/SVD.Solvers.fProxy.cs` — 3 `pinvSolve` + 3 `pseudoInverse` `ref fProxySvd_WS ws`
      overloads (full, relTol-only→maxSweeps 30, none→relTol -1/maxSweeps 30), forwarding to the
      scratch primitive. Disambiguated from the `relTol`/`maxSweeps` overloads by the 5th-arg type.
- [x] Tests: `SVDWorkspaceTests` augmented — ws path ≡ raw-scratch path (pinv/pseudo, tall+wide) +
      `SvdWorkspace_Factory_SizesCorrectly` + `WorkspaceReuse` (one ws reused across 3 solves ≡ fresh
      alloc each time). SVD suite 117/117; full suite 1083/1083 (pre-reuse-test).
- [x] bug-hunter audit of the workspace API: CLEAN (overload resolution, method==type name, ref-field
      forwarding, factory sizing, default-At guard all sound). Flagged the reuse-coverage gap → CLOSED.

## Status: PHASE-2 COMPLETE

Every solver/decomposition with an internal scratch buffer now has a caller-provided-workspace
overload (primitive) + allocating wrapper. Confirmed by grep: the only internal workspace allocs
in the OP templates were the Ortho_OP QR `new fProxyN(...Allocator.Temp...)` (now wrappers). LU,
Cholesky, Eigen, and the SVD core (`svdDecomposition`) were already zero-alloc (operate on
caller-provided `ref` outputs); CG was done in the phase-1.5 batch.
- [x] Tests: `OrthoWorkspaceTests.fProxy.cs` — QR scratch≡alloc equivalence + mis-sized guard +
      solveQR solve(square/tall) + solveQR alias-b/bad-size guards. (SVD tests pending with chunk 3.)

Done = every box checked, suite green, allocating forms delegate to scratch forms.

## Notes / pitfalls

- QR scratch `u` MUST be length `M_Rows` exactly: `genHouseholderPete` iterates `r < u.N` and
  reads `Q[r, k]`, so an oversized `u` would read `Q` out of bounds. Hence exact-size guard.
- The allocating wrappers must keep `[MethodImpl(AggressiveInlining)]` parity with the originals
  and preserve the original allocator (`Allocator.Temp`) so behavior is unchanged for existing callers.
