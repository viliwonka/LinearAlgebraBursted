# Spec: SVD/Eigen convergence overhaul — status structs + sweep budgets

Status: APPROVED 2026-07-06 (user: SVD should return struct results like the other
decomposers). Two coupled parts, one change: (A) size the iteration budgets so legitimate
inputs never hit them; (B) replace ignorable `bool` returns with status structs per the
library-wide diagnostics convention.

## Motivation

2026-07-06 finding: `SVD.thin` in double on a clustered spectrum (randsvd 0.95-decay,
n=512) exhausts the default sweep budget and returns `false` — which a caller can silently
drop, then consume an untouched (zero) S. Two defects: the budget default is too small for
legitimate clustered-double workloads (LAPACK sizes budgets so they never fire on real
input), and `bool` is the only return in the library that violates the "all solvers return
a status struct" convention (DirectSolveInfo / RankInfo / SolveInfo / EigenSolveInfo /
LanczosInfo / LOBPCGInfo everywhere else).

## Part A — sweep budgets (RESOLVED: adopt LAPACK's scaling, battery VERIFIES it)

Current semantics (SVD.fProxy.cs bidiagonalQR, and the Eigen QL sweeps): `maxIter` is
PER SINGULAR VALUE / per eigenvalue (NR-style `for k { for its < maxIter }`), default 75.
Failure observed: a 5%-spaced 512-value cluster in double needs >75 sweeps for at least
one value.

USER DECISION 2026-07-06: don't derive the constant by measurement — copy the known
literature value. LAPACK dbdsqr budgets MAXITR=6 as a TOTAL of 6·n² iterations, i.e.,
an effective 6·n per value — LAPACK's budget SCALES WITH n (that scaling is exactly what
our flat NR-style constant lacked).

1. New default, per-value semantics kept: **maxIter_default = max(75, 6·n)** computed at
   the convenience-overload layer (n = the relevant dimension per method). Explicit
   maxIter args keep their current per-value meaning — no semantic migration.
2. The budget battery (graded 0.95/0.99 decay, clustered-with-gaps, random; n ∈ {128,
   512, 1024, 2048}; float+double; thin/values + Eigen.valuesSymmetric/symmetric) is a
   VERIFICATION gate: every case must return Success with the new default AND report
   max-sweeps-per-value observed (via the new SVDInfo.sweeps) ≤ ~1/4 of the budget.
   If any case violates the margin, STOP and report — do not quietly bump the constant.
3. Battery may run cases as PARALLEL scheduled jobs (user-approved): no timing is being
   measured, only convergence/accuracy, so throughput-parallelism is fine — one Unity
   invocation, many jobs.
4. The cap stays a pathological-input backstop only; document that in the XML.
5. Audit every flat-75 forwarding overload found by grep (thin, values, truncated,
   randomized, pinvSolve, nullspaceBasis, rangeBasis) + Eigen QL caps — all move to the
   scaled default.

## Part B — status structs (RESOLVED per user 2026-07-06)

New struct (Solvers.Info.cs, house pattern):

```
SVDInfo {
  IterativeSolveStatus status;   // Success / MaxIterations (reuse the existing enum —
                                 // no new per-family enum, per the diagnostics decision)
  int sweeps;                    // total QR/QL sweeps consumed (already a loop counter)
  int converged;                 // count of values that converged (== n on success;
                                 // LOBPCGInfo.converged count precedent)
  public static implicit operator bool(SVDInfo i) => i.status == Success;
}
```

The **implicit bool operator** is user-required: `if (SVD.thin(...)) { ... }` must keep
working for callers who only care about success. (Check whether Burst tolerates implicit
operators on the struct — they're plain static methods, should be fine; if not, a
`.Solved`-style bool property is the fallback, matching DirectSolveInfo, and report the
deviation.) Fields obey the "already-computed or trivially cheap" rule. NO residual field
(not computed; that is what the test oracles are for).

Twin struct **`EigenInfo`** — same shape, own type (user decision OQ-2: copy, don't
share) — returned by the Eigen family. If the name collides with anything existing
(EigenSolveInfo stays for power/inverse iteration — different shape, keeps its role),
report before improvising.

Surface changes (hard replace, no shims — pre-release policy):
- `SVD.thin`, `SVD.values` (all overloads): `bool` → `SVDInfo`.
- `SVD.truncated`, `SVD.randomized`: fold `out bool converged` into the returned
  `SVDInfo` (drop the out param).
- `SVD.pinvSolve`: returns `int rank` + `out bool converged` today → return **`RankInfo`**
  (exists; rank + status is exactly its shape; status maps non-convergence appropriately —
  see OQ-1).
- `nullspaceBasis` / `rangeBasis`: `int dim` + `out converged` → `RankInfo` (dim is
  derivable: n − rank / rank; keep an int return ONLY if call-site ergonomics degrade
  badly — implementer judgment, report it).
- `Eigen.symmetric`, `Eigen.valuesSymmetric`, `Eigen.valuesQR`, `Eigen.decompInPlace`:
  `bool` → `EigenInfo` (same shape + implicit bool).
- `Bidiag.decomp/values`: finite/direct, void/no convergence — UNCHANGED.
- Matrix-free eigensolvers (powerIteration/lanczos/LOBPCG): already struct-returning —
  UNCHANGED.
- `PCA.fitSvd/fitSvdTruncated/fitRandomized` (RESOLVED OQ-3): keep `bool` return
  (true ONLY if the underlying SVD converged) + add **`out SVDInfo info`** discardable
  with `out _`. `fitCov` (eigen route): same pattern with `out EigenInfo` — the one
  spot where the copied-twin decision shows; acceptable, note it in docs.

Failure contract text (XML, uniform): "On MaxIterations the outputs are NOT usable:
S/U/V (or eigenvalues/vectors) are unwritten or partial; check the returned status
before use." (Matches the S-written-only-inside-ok behavior verified by the commit-2.5
tests.)

## Tests

- Mechanical: every existing bool-checking call site updated (`Assert.IsTrue(ok)` →
  status checks) — do NOT weaken any assertion.
- New: non-convergence regression — the discovered case (randsvd 0.95 decay, n=512,
  double) run with a deliberately tiny explicit maxIter, asserting
  status==MaxIterations, unconverged>0, and S untouched (NaN-sentinel prefill).
- New: budget-default adequacy — the same case with DEFAULT budget must return Success
  (this is the Part-A acceptance test).
- sweeps field sanity: Success ⇒ sweeps>0; monotone-ish vs n on the budget battery
  (loose assert, it's a diagnostic not a contract).

## Open questions

- **OQ-1** (still open, implementer resolves + reports): `RankInfo.status` enum member
  for non-convergence — does the enum RankInfo carries have a suitable member, or does
  one get added? Adding an enum member is fine; adding a new enum is not.
- **OQ-2**: ✅ RESOLVED — copy the struct (twin `EigenInfo`), no sharing.
- **OQ-3**: ✅ RESOLVED — PCA fit* keeps `bool` return (true only if the decomposition
  converged) + gains `out SVDInfo`/`out EigenInfo` discardable with `out _`.
- **OQ-4** (still open, implementer verifies): `Analysis.cond`/`rank`/`matrixL2` (route
  through values) — RECOMMEND no surface change; they handle non-convergence internally
  today; verify and keep.

## Pipeline

Serialized AFTER the QRCP downdate task (Unity project lock). coder → adversarial review
(focus: no assertion weakened in the bool→struct sweep; failure contract actually holds
on every family member) → full suite → commit (not push).
