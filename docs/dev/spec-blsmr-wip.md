# blsmr (Block LSMR) — WIP handoff

Status: **PARKED, not integrated.** All blsmr files were removed from `Assets/` (they would
otherwise break the user's Unity build — Unity compiles untracked files too) and moved to
`reference/wip-blsmr/`:

- `reference/wip-blsmr/Krylov.Block.LSMR.fProxy.cs` — the template (core algorithm + dense
  forwarders), was `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.LSMR.fProxy.cs`.
- `reference/wip-blsmr/BlockLSMRTests.fProxy.cs` — the bespoke test file, was
  `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BlockLSMRTests.fProxy.cs`.

`git status` / `Tools/regen.ps1` confirm main is clean: no blsmr files anywhere under `Assets/`,
codegen reports "generated files are in sync with the templates" (no orphans, no drift).

## What's implemented

The template implements the full Bl-LSMR algorithm (Mojarrab & Toutounian 2015, Algorithm 2 +
Block Bidiag 1) per `reference/rectangular/BlockLSQR-LSMR-algorithm-extract.md`, ported to this
codebase's row-major block-RHS convention (`fProxyMxN` s rows x length cols, row j = RHS/solution
vector j — same convention as `bcgrq`/`bidr`/`bminres`).

Key derivation decisions (see the template's own doc comments):
- The paper's column-oriented thin QR of the block bidiagonalization (`U_i B_i`, `V_i A_i`) becomes,
  under this codebase's ROW-major block storage, a thin **LQ** decomposition (`LQ.decomp`, the
  UNPIVOTED one — NOT `LQRP`, whose row-pivoting would silently scramble the block-Krylov basis
  correspondence; this was verified carefully, see the file's own comments). The paper's "A_i"/"B_i"
  upper-triangular R-factors map to this codebase's `LA`/`LBnew` etc. LQ **L**-factors via
  `L := (paper factor)^T` — the transpose falls out for free because LQ's L is already the transpose
  of the equivalent column-oriented QR's R.
- The two nested block-Givens rotation stages (the block generalization of scalar LSMR's
  `rho/c/s` then `rhobar/cbar/sbar`) are realized by QR-factoring the stacked `[alphadotk; Bbark+1]`
  (2s x s) padded to a **square 2s x 2s matrix with a zero right half**, then routing through the
  existing (unpivoted, full-Q) `QR.decomp` — the padding trick avoids needing a dedicated
  "full/completed Q from a tall QR" primitive (the library's QR only exposes the thin Q otherwise).
  `Qfull`'s first-s columns / `Rfull`'s top-left s x s block reproduce the true thin QR of the
  s-wide input exactly (Householder reflectors for columns 0..s-1 never see the padded zero columns);
  `Qfull`'s second-s columns give a valid (if not uniquely pinned by the paper) orthogonal
  completion for the `[cbark dbark]` block-rotation row.
- Breakdown = the paper's own stated condition (`alphabark` singular) plus the same check on every
  block-bidiagonalization LQ factor (`TriNearSingular`: any collapsed diagonal entry relative to the
  largest).

## What's NOT verified — the actual blocker

The algorithm's correctness could NOT be confirmed inside a real `[BurstCompile(CompileSynchronously
= true)] IJob` in the time available. Investigation trail:

1. First pass: the bespoke tests (`NormalEquationsOptimalAndMatchesScalarLsmr`,
   `ConsistentSystemRecoversExactSolution`, `ZeroRhsConvergesImmediately`, `NeverNaNOnTinyMaxIter`),
   run via `Tools/run-tests.ps1 -Filter "*BlockLSMR*"`, reported `Result=Passed total=15 passed=15
   failed=0` — looked green.
2. Added debug output fields to `BlockSolveInfo`-adjacent job fields and discovered
   `ConsistentSystemRecoversExactSolution` was actually returning `status=Converged, iterations=0`
   — i.e. hitting blsmr's very first `‖AᵀB‖²_F == 0` early-out, meaning **B itself was reading back
   as all-zero inside the Burst job**, not that the algorithm was solving the system.
3. Added a `[BurstDiscard]`-guarded debug print plus a NEW plain (non-`[BurstCompile]`, direct Mono
   call, no `IJob.Run()`) test method (`DebugNoBurst`) that builds the identical A/B/Xk (same fixed
   seeds) and calls `Krylov.blsmr` directly. **This worked correctly**: `atbSqF≈456.8` (matches an
   independently-computed `‖AᵀB‖²_F`), `status=Converged, iterations=2, maxRnorm≈1e-6` for both
   float and double — i.e. the ALGORITHM looks right when driven outside Burst.
4. Added a matching in-Burst-job diagnostic (`DebugInBurst`) that replicates the exact same
   construction + a manual `‖AᵀB‖²_F` computation using ONLY public APIs (`Blas.dot` per row via a
   `Row()`/`arena.fProxyVec()` helper — the SAME pattern `BlockCGrQTests.fProxy.cs`'s own `Row()`
   helper uses, which is known to work in that file's already-green tests). Result: **`aSqF = bSqF =
   xkSqF = atbSqF = 0`** — i.e. even `A` and `Xk` themselves (built via
   `arena.fProxyRandomMat`, nothing blsmr-specific) read back as all-zero when read INSIDE this
   particular Burst job, despite the identical code working fine under Mono.

This means the "15/15 passed" result in step 1 is very likely a **false positive**: if A/B/Xk are
all silently zero inside the job, `X = 0` trivially satisfies every assertion in every test
(normal-equations residual is `‖Aᵀ(0·0 - 0)‖ = 0`; the "consistent system" and "known solution"
checks degenerate to `0 == 0`). The test suite was NOT actually exercising blsmr's arithmetic.

**Root cause not identified.** Candidates not yet ruled out, in order of suspicion:
- Something specific to this test job's shape (many sequential `Arena`-allocated matrices +
  `arena.fProxyRandomMat` calls + a `blsmr` call all inside one `Execute()`) interacting badly with
  Burst — since `BlockCGrQTests`' own `Row()`/arena pattern is presumably fine in isolation (that
  suite is part of the green baseline), the trigger may be dosage/ordering/size specific, not the
  pattern per se.
- An arena chunked-record-table staleness issue specific to Burst-compiled code (`AssertRecordValid`
  and similar safety-check-gated logic behaves differently when
  `ENABLE_UNITY_COLLECTIONS_CHECKS`-style checks are compiled out).
- Something in how `Unity.Mathematics.Random`-seeded `fProxyRandomMat` behaves under Burst for this
  specific call sequence (less likely — same seeds, same call shape as many other green block-solver
  tests elsewhere in the suite).

This is NOT confirmed to be a bug in `blsmr`'s own algorithm — it looks environmental/harness-side,
but that couldn't be confirmed either given the time box.

## To resume

1. Move the two files back from `reference/wip-blsmr/` to their original `Assets/...` paths, run
   `Tools/regen.ps1`.
2. FIRST, before trusting any test result again: reproduce the `DebugInBurst`-style diagnostic (a
   `[BurstCompile(CompileSynchronously = true)] IJob` that builds a small `fProxyRandomMat` A/B via
   `Arena` and reads a checksum like `sum(A[i,j]^2)` back out) in ISOLATION, decoupled from blsmr
   entirely, to nail down whether this is a general Arena-in-Burst-job hazard (in which case it
   likely affects other things and is worth its own investigation/fix) or something specific to this
   test file/job shape.
3. Once test data reads back correctly, rerun `NormalEquationsOptimalAndMatchesScalarLsmr` and
   `ConsistentSystemRecoversExactSolution` for real signal. If they pass with real (nonzero) data,
   this is very likely done — the algorithm and derivation were exercised via `DebugNoBurst`
   (non-Burst) and looked correct (fast convergence, small residual, both dtypes).
4. Only if the algorithm turns out wrong once given real data: re-derive Bbark's/the block-Givens
   completion's correctness — start from the `Qfull` second-half-columns non-uniqueness caveat noted
   in the template's own comments as the most likely mathematically-soft spot (see "What's
   implemented" above); everything else was cross-checked line-by-line against
   `BlockLSQR-LSMR-algorithm-extract.md` and dimensionally verified.
5. Not yet added (skip until green): a BSR overload (trivial one-line forwarder once the dense path
   is confirmed, mirrors `lsmr(in fProxyBSR A, ...)`), and wiring into any battery (there is no
   block-least-squares battery family yet — out of scope per the original task spec anyway).

## Reference deviations from the spec

- No warm-start support (`X0 = 0` fixed, matching the paper's Algorithm 2 exactly) — scalar `lsmr`
  supports warm start via residual bidiagonalization, blsmr does not. Worth revisiting if warm start
  is ever needed for blsmr; the paper's algorithm as extracted doesn't cover it.
- No Tikhonov damping parameter (the paper's Bl-LSMR has none; scalar `lsmr` has `damp`).
- Uses `IfProxyLinearOperator.Apply`/`ApplyT` per-row through a `[private] BlockApplyOp`/
  `BlockApplyOpT` helper (a Temp-row-buffer loop), NOT a fused block GEMM — `ApplyBlock` on the
  interface is documented as symmetric/square-only and doesn't fit a rectangular A. This is the same
  fallback shape `fProxyColScaledOperator.ApplyBlock` already uses elsewhere. Correct but not the
  fastest possible; a dense-A-specific fused path (mirroring `fProxyDenseOperatorGeneral`) would be a
  reasonable follow-up once correctness is confirmed.
