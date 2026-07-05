# Pre-Public-Release Readiness Audit

*Historical document — method names predate the 2026-07 solver-API rework (see
docs/spec-solver-api-rework.md for the mapping).*

Verified against the actual repo (not just memory). **Bottom line: the numerical CORE is
release-ready; the gaps are almost entirely release ENGINEERING and presentation.**

## BLOCKERS (fix before a "final public release")
- **B1 — README says not ready.** `README.md:4`: "experimental... not yet ready for production use."
  Contradicts a final release; it's the first thing a visitor reads. Rewrite the positioning; state the
  supported Unity version (6000.3.2f1).
- **B2 — Not a Unity package.** No UPM `package.json` under `Assets/` (only the third-party codegen
  pkg). No git tags. Install is "copy `Source/` into your project." GOOD NEWS: `Source/` is
  self-contained (259 committed generated `.cs`; runtime asmdef deps = Unity.Mathematics/Burst/
  Collections only, NOT the codegen pkg) → it CAN ship as a package today. Add
  `Assets/LinearAlgebra/package.json` + semver + a `Samples~/`, cut a `v0.x.0` tag.
- **B3 — README quick-start doesn't compile.** `README.md:69` uses `Norms_OP.L1(...)`; the generated
  type is per-type `floatNorms_OP`/`doubleNorms_OP` — there is no `Norms_OP`. Copy-paste → compile
  error. Fix the example (ideally compile it as a Sample so it can't drift).
- **B4 — No CHANGELOG / CONTRIBUTING.** Neither exists. LICENSE (MIT) is present & correct. Add a
  Keep-a-Changelog `CHANGELOG.md` and a short `CONTRIBUTING.md` documenting the codegen workflow
  (templates are source of truth; `Tools/regen.ps1`, `Tools/run-tests.ps1`).

## SHOULD-FIX
- **S1 — No CI.** No `.github/workflows`. Only a local `Tools/git-hooks/pre-push`. Add a game-ci
  EditMode workflow on PRs (+ a badge — matters for a numerical library).
- **S2 — Dead "View" code shipped public.** Views were DROPPED (README 152-154) yet
  `struct viewBoolMxN` ships with comment `//todo: check correctness, idk what it does`
  (`Source/Generated/View/MatView.bool.cs:12,66`). Delete the `View/` template + generated output.
- **S3 — `NotImplementedException` in public interface.** `floatMxN`/`doubleMxN`
  `IMatrix<T>.CopyTo`/`CopyFrom` throw NotImplementedException (`Generated/float/floatMxN.cs:112-118`).
  Implement (trivial) or remove from the interface.
- **S4 — XML-doc coverage ~67% (351/527), uneven.** Core factorizations fully documented, but
  **QR 3/12** and **Solvers 6/11** — the most-used entry points are the thinnest. Fill those first; add
  numerical caveats (SPD for Cholesky, rank-deficiency behavior for LS/min-norm).
- **S5 — Two arena-temp allocs (prior perf audit, still unfixed, low impact).**
  `Realtime/RollingWindow.fProxy.cs:181`, `Statistics/StatsOP.fProxy.cs:652`. Switch to a disposed
  `Allocator.Temp` or document as intentional.
- **S6 — Thin tests on some shipped features.** Optimize (1 case), K-Means (7, no convergence/
  empty-cluster), Gallery/Histogram/Resample (4-5 each). Suite is otherwise strong (~1460 [Test] →
  ~3050 parameterized, with real known-answer/literature tests). Add convergence/edge-case tests for
  Optimize and K-Means before advertising them.

## NICE-TO-HAVE
- **N1** No user-facing performance docs (internal benchmarks are excellent). Add a README
  "Performance" section (GEMM ~70 GFLOP/s reference, the zero-alloc workspace pattern, now the blocked
  QR/LQ).
- **N2** `[Obsolete]` Jacobi eigen/SVD retained as reference — fine; exclude from the "supported" list.
- **N3** API-freeze decisions: `operator *` is element-wise Hadamard (`fProxyMxN.Operators.cs:150`) —
  document loudly or reconsider; `M_Rows`/`N_Cols` are mutable public fields
  (`fProxyMxN.cs:12-13`) — desync risk, consider get-only.
- **N4** Cosmetic micro-nits (`Eigen.fProxy.cs:288-289` hot-loop branch, `:975` dead no-op).

## VERIFIED HEALTHY (do not re-open)
- All four named 2026-06-28 correctness bugs are FIXED (acosh, `^n` off-by-one, bool null-guard);
  DingDong was a false alarm. All HIGH-severity perf findings fixed.
- Exception convention consistent (`ArgumentException`/`ArgumentOutOfRangeException`, no bare
  `Exception`, static messages) and documented in `docs/naming-style-guide.md`.
- Naming refactor is coherent (the sub-agent's three "blockers" were false positives against the
  documented convention).

## Suggested ordering
B1–B4 (≈ a day; unblocks shipping) → S1 (CI) / S2 (delete View) / S3 / S4 (QR+Solvers docs) →
S6 test gaps → nice-to-haves.
