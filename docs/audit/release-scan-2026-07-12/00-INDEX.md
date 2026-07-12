# Release bug scan — 2026-07-12 — index

Full-library audit of the template trees (source of truth): `TemplateSource` (core),
`TemplateSourceTests`, `TemplateSourceBenchmarks` — ~124k lines, 380 files, 22 areas.

Method: one scanner agent per area read every file in full, hunting five bug classes
(naming/comment-vs-behavior, numerical, performance, logical, pointer/memory). Every
finding was then adversarially verified by an independent agent instructed to refute it.
140 agents total; verdicts: **CONFIRMED** (failure traced), **UNCERTAIN** (plausible, not
provable from code alone), **REFUTED** (claim wrong — kept in each report's appendix).

## Totals

- 96 findings: **92 confirmed**, 1 uncertain, 3 refuted.
- Severity (non-refuted): **1 high**, **23 medium**, 69 low.
- Production-code (core) share: 1 high, 9 medium; the rest are in tests/benchmarks.

## The one HIGH

- `OP/OP.Dot.fProxy.cs:278` — **householderInPlace does not apply a Householder
  reflection.** It subtracts a constant outer product (2/uᵀu)·u·uᵀ that ignores the
  matrix contents, so H·M is wrong for any M ≠ I. (Report 02.)

## Confirmed MEDIUM — core (9)

| Where | Class | Finding |
|---|---|---|
| `OP/UnsafeOP.iProxy.cs:115` | performance | iProxy `vecMatDot` uses cache-hostile column-strided inner loop (fProxy sibling was optimized, iProxy never was) |
| `OP/Bidiag.fProxy.cs:237` | pointer | `Bidiag.decomp` convenience overload allocates 5 Temp buffers before dimension validation → leak on validation throw |
| `OP/Bidiag.fProxy.cs:326` | pointer | same pattern in `Bidiag.values` (4 buffers) |
| `OP/SVD.fProxy.cs:37` | logical | `tolerance` param of `SVD.values`/`SVD.thin` is validated but never used — deflation threshold hardcoded to `Consts.fProxyEpsilon*anorm` |
| `OP/MIP.fProxy.cs:364` | logical | node budget checked after LP solve but before incumbent extraction — last permitted node's integer-feasible solution discarded (`maxNodes=1` on integral root LP returns NodeLimit, no incumbent) |
| `Pivot/Pivot.Operations.cs:121` | pointer | `ApplyInverseVec/Row/Column` lack the dimension guard their forward counterparts have → buffer overflow on size mismatch |
| `Arena/ArenaExtensions.fProxy.cs:240` | numerical | `fProxyHouseholderMat` divides by vᵀv with no near-zero guard → NaN/Inf for zero/tiny vector |
| `Sparse/Norms.Sparse.fProxy.cs:12` | numerical | L1 under-counts Symmetric-storage BSR while doc claims equality with dense expansion |
| `Sparse/Norms.Sparse.fProxy.cs:33` | numerical | L2/Frobenius claimed "exact" but under-counts Symmetric-storage off-diagonals |

## Confirmed MEDIUM — tests (14)

Recurring themes: **vacuous tests** (empty bodies, tautological asserts, missing [Test]
wrappers, switch cases that don't exist), **blocked-QR gates never reached** (tests use
size 64 but the real gates are 128/512), and **arena leaks in tests**.

| Where | Finding |
|---|---|
| `fProxy/DotOperationTests:169` + `iProxy:169` | self-dot computes D = dot(C,C) but asserts on C — product never verified |
| `fProxy/DotOperationTests:217` + `iProxy:217` | `MatMatDotNonSquare` body is empty — the [Test] passes vacuously |
| `fProxy/SpecialConstructorsTests:79` | [Test] RandomRangeMat runs a job whose switch has no such case — exercises nothing |
| `fProxy/SpecialConstructorsTests:306` | `-math.abs(...) < eps` is always true — off-diagonal never validated |
| `fProxy/SpecialConstructorsTests:190` | IndexZeroMat/IndexOneMat leak a Persistent Arena |
| `fProxy/IndexingTests:157` + `iProxy:162` | RandomCalc leaks a Persistent Arena every run |
| `fProxy/InitTest:26` | job calls `arena.Clear()` but never `Dispose()` — arena core leaks |
| `fProxy/ConjugateGradientTests:281` | SingularConsistent case implemented but has no [Test] wrapper — never runs |
| `fProxy/QRTests:318` | "blocked non-aligned" tests never reach the blocked kernel (gate is 128/512, not 64; double build entirely unblocked) |
| `fProxy/QRCacheWorkspaceTests:13` | blocked compact-WY cache-equivalence tests never reach the blocked path |
| `fProxy/CompMathTests:382` | comment claims test FAILS due to a live kernel bug — false/stale, plus embedded reviewer note |

## Per-area reports

| # | Report | Files | Confirmed | H/M/L (non-refuted) |
|---|---|---|---|---|
| 01 | [comp-unsafe-ops](01-comp-unsafe-ops.md) | 17 | 3 | 0/1/2 |
| 02 | [dot-blas-simd](02-dot-blas-simd.md) | 11 | 4 | **1**/0/3 |
| 03 | [lu-cho-bidiag](03-lu-cho-bidiag.md) | 6 | 4 | 0/2/2 |
| 04 | [qr-lq](04-qr-lq.md) | 9 | 4 | 0/0/4 |
| 05 | [svd](05-svd.md) | 12 | 3 | 0/1/2 |
| 06 | [eigen-krylov](06-eigen-krylov.md) | 10 | 4 | 0/0/4 |
| 07 | [lp](07-lp.md) | 9 | 5 | 0/0/5 |
| 08 | [qp-mip](08-qp-mip.md) | 7 | 3 | 0/1/2 |
| 09 | [control-signal](09-control-signal.md) | 10 | 2 | 0/0/2 |
| 10 | [random-query](10-random-query.md) | 17 | 4 | 0/0/4 |
| 11 | [types-core](11-types-core.md) | 27 | 2 | 0/0/2 |
| 12 | [types-int-bool](12-types-int-bool.md) | 24 | 1 (1 refuted) | 0/1/0 |
| 13 | [arena](13-arena.md) | 18 | 7 (1 refuted) | 0/1/6 |
| 14 | [sparse](14-sparse.md) | 20 | 5 | 0/2/3 |
| 15 | [ml-stats-debug](15-ml-stats-debug.md) | 27 | 4 | 0/0/4 |
| 16 | [tests-shared-iproxy](16-tests-shared-iproxy.md) | 35 | 6 | 0/3/3 |
| 17 | [tests-f-a-d](17-tests-f-a-d.md) | 22 | 5 (1 refuted) | 0/4/1 |
| 18 | [tests-f-e-l](18-tests-f-e-l.md) | 24 | 4 | 0/2/2 |
| 19 | [tests-f-m-r](19-tests-f-m-r.md) | 18 | 5 | 0/2/3 |
| 20 | [tests-f-sparse](20-tests-f-sparse.md) | 14 | 5 | 0/0/5 |
| 21 | [tests-f-s-v](21-tests-f-s-v.md) | 19 | 5 | 0/3/2 |
| 22 | [benchmarks](22-benchmarks.md) | 24 | 7 (1 uncertain) | 0/0/8 |

## Non-template sweep (second pass, same day)

Codegen infrastructure, Tools scripts, hand-written tests/benchmarks, demos, and
user-facing docs (fact-checked against the API, prose untouched). 26 confirmed
(1 high, 5 medium, 20 low), 1 refuted. The high + all mediums were fixed the same
day: TemplateConverter caps-bool token (latent), regen.ps1 -Check scope,
benchmark/run-tests Get-Content UTF-8 decode, control.md/decompositions.md
placeholder type names, print-export.md stale LOBPCGInfo claim.

| # | Report | Files | Confirmed | H/M/L (non-refuted) |
|---|---|---|---|---|
| 23 | [codegen-infra](23-codegen-infra.md) | 6 | 3 | 0/1/2 |
| 24 | [tools-scripts](24-tools-scripts.md) | 8 | 5 | 0/2/3 |
| 25 | [bench-handwritten](25-bench-handwritten.md) | 27 | 3 | 0/0/3 |
| 26 | [tests-handwritten](26-tests-handwritten.md) | 11 | 3 | 0/0/3 |
| 27 | [demos](27-demos.md) | 9 | 4 (1 refuted) | 0/0/4 |
| 28 | [docs-userfacing](28-docs-userfacing.md) | 19 | 8 | 1/2/5 |

## Post-scan code sweep (third pass, same day)

Everything shipped after the main audit: the Kalman family (KF/EKF/UKF/LQG),
condensed MPC + the QP warm-start seam, NLS/curve fitting, their tests, demos
09-11, and the new benchmark harnesses. 15 findings, 10 confirmed (2 high),
5 refuted. All confirmed findings fixed the same day. The MPC high (prestabilized
input-bound rows off by one block) led, via the equivalence-oracle test written
for the fix, to a second latent defect (Hessian missing the R cross-coupling term
in prestabilized mode) — both fixed through one shared affine map, with a
discriminating prestab-vs-raw equivalence test added to the suite.

| # | Report | Files | Confirmed | H/M/L (non-refuted) |
|---|---|---|---|---|
| 29 | [kalman-family](29-kalman-family.md) | 8 | 1 | 0/0/1 |
| 30 | [mpc-qpseam](30-mpc-qpseam.md) | 4 | 2 | 1/0/1 |
| 31 | [nls](31-nls.md) | 3 | 2 | 0/0/2 |
| 32 | [new-tests](32-new-tests.md) | 4 | 1 (2 refuted) | 0/0/1 |
| 33 | [new-demos](33-new-demos.md) | 6 | 2 (2 refuted) | 1/0/1 |
| 34 | [new-benchmarks](34-new-benchmarks.md) | 4 | 2 (1 refuted) | 0/0/2 |

Low-severity findings (69) are mostly doc/comment contradictions and comment-policy
violations; see each report. Fixes should be made in the templates and regenerated
(`Tools/regen.ps1`), never in `Assets/LinearAlgebra/Source`.
