# Dense types & the Arena allocator

`floatN`/`floatMxN` and their `double`/`int`/`short`/`long`/`uint`/`bool` counterparts are the
library's vector and matrix types. Matrices are row-major (`Data[r*N_Cols+c]`) — the opposite of
`Unity.Mathematics`' column-major layout, so any conversion between the two is a transpose, not a
reinterpret-cast; see [spec-interop.md](../spec-interop.md)'s "Row-major ↔ column-major" section for
the full correctness argument.

## Arena

`Arena` is a cheap-to-copy handle (a pointer to a heap-allocated core) to a bump/tracking allocator.
All copies of an `Arena` share one core, so passing it by value is safe — but **disposing two copies
of the same handle double-frees**; treat an `Arena` like a single owned resource.

- `new Arena(Allocator allocator)` — `Allocator.Persistent` for long-lived state, `Allocator.Temp`
  for a single frame/job.
- `arena.floatVec(int N, bool uninit = false)` / `arena.floatVec(int N, float s)` — vector factories
  (zeroed, or filled with `s`). `arena.floatMat(int rows, int cols, ...)` — matrix factories, plus a
  square `floatMat(int dim)` overload.
- `ArenaExtensions` adds everything else as `this ref Arena` extensions: `floatIdentityMat`,
  `floatRandomMat`, `floatRandomDiagonalMat`, `floatBasisVec`, `floatLinVec`, `floatHouseholderMat`,
  `floatHilbertMat`, and per-feature workspace factories (`floatSVDThinCache`, `floatFFTCache`, …).
- `arena.ClearTemp()` — disposes only the **temp** pool (the scratch every allocating op/operator
  produces); call it once per frame/loop iteration to keep temp allocations from accumulating.
- `arena.Dispose()` — disposes everything (persistent + temp) and frees the core itself.

**Threading contract (from the type's own doc comment):** an `Arena` is single-threaded by contract
— like Unity's native containers, a single instance must not be mutated (allocated from, cleared, or
disposed) from more than one thread/job concurrently. Concretely:

- **Allocate before you schedule.** Do an arena's persistent allocations (and any `Pivot`/`Indices`
  buffers a factorization needs) on the scheduling thread before handing data derived from it to a
  job. A job's `Execute()` can itself call arena factories/`Clear`/`ClearTemp` (they're
  Burst-compilable), but only if no other thread is touching the same arena at the same time.
- **One arena per concurrently-running job/thread.** If two jobs may run at the same time (no
  dependency between them), give each its own `Arena`. Sharing one arena across concurrent jobs
  races the record tables' chunk-directory/free-list bookkeeping — silently, with no exception, just
  wrong answers or a crash later.
- **Complete jobs before `Clear()`/`ClearTemp()`/`Dispose()`.** Wait on a job's `JobHandle` before
  tearing down (or clearing) an arena it might still touch.

### Concurrency guards (detection, not prevention)

Nothing about `Arena`/`ArenaCore` makes concurrent misuse safe — the contract above still has to be
followed. What exists is *detection*: under `ENABLE_UNITY_COLLECTIONS_CHECKS`, every mutating entry
point (the `arena.floatVec`/`floatMat`/... factories, `fProxyBSR`/`fProxyBSRBuilder`/
`fProxyBlockJacobi`, `Clear`, `ClearTemp`, `Dispose`, `Pivot`, `Indices`) is wrapped by two
independent mechanisms so a contract violation throws instead of corrupting memory silently:

1. **Race tripwire.** An `int` flag inside `ArenaCore` (heap-resident, so it doesn't affect
   `sizeof(Arena)`) is armed via `Interlocked.CompareExchange` on entry to a guarded body and
   released via `Interlocked.Exchange` on exit. If a second thread/job enters a guarded body while
   the flag is already armed, it throws `InvalidOperationException` immediately — this is the
   mechanism that actually catches two jobs racing the same arena. It is not a mutex: it never
   blocks, it only detects overlap. Reentrancy (one guarded method legitimately calling another,
   e.g. `Clear()` internally clearing the temp pool too) is handled structurally, by routing
   internal calls through unguarded "core" helpers instead of nesting the guard — not by counting,
   since Burst has no `Thread.CurrentThread` to distinguish legitimate same-thread nesting from a
   genuine second thread.
2. **`AtomicSafetyHandle`.** `ArenaCore` also owns an `AtomicSafetyHandle`, created in `Init` and
   released in `Dispose`, checked at the top of every guarded entry point. This catches a live
   handle used (from any thread) after `Dispose()` already released it.

**Why not Unity's `[NativeContainer]` job-scheduling protocol** (the mechanism that makes the job
debugger reject two `IJob`s capturing the same container without a dependency, *at Schedule time*)?
That protocol requires the `AtomicSafetyHandle` to be a field directly on the struct a job captures
by value — Unity's own `NativeList<T>`/`NativeReference<T>` carry `m_Safety` inline for exactly this
reason. The struct a job captures here is `Arena`, which is deliberately pinned to a single
pointer's width (`ArenaLayoutTests.Arena_IsPointerSized`) so it stays cheap to copy and pass around;
adding an `AtomicSafetyHandle` field to it would grow `sizeof(Arena)` under
`ENABLE_UNITY_COLLECTIONS_CHECKS` and break that pin. The handle instead lives inside the
heap-allocated `ArenaCore` that `Arena` points to, which keeps `Arena` pointer-sized but means there
is no automatic Schedule-time rejection — only the manual checks above, which fire once the
conflicting code actually runs, not the moment it's scheduled.

**Known gap:** an individual buffer's own `Dispose()` (e.g. `floatN.Dispose()`) also mutates a
record table (frees its slot) but is *not* guarded by the tripwire — it sits at a different altitude
than the Arena/ArenaCore entry points above (every numeric/bool/sparse type's own per-instance
lifecycle method, closer in spirit to "element access"). Two threads disposing different allocations
from the same arena concurrently is therefore still a real, undetected race.

## Vectors & matrices

Both types carry either an arena-tracked record or a standalone `UnsafeList` (for `new floatN(n,
Allocator.Temp)`-style construction outside an arena) behind one indexer surface:

- Indexers: linear `this[int]` / `this[System.Index]` (from-end supported), and for matrices
  `this[int r, int c]` (bounds-checked only under `ENABLE_UNITY_COLLECTIONS_CHECKS`).
- Fields: `floatMxN.M_Rows`, `.N_Cols`, `.Length`, `.IsSquare`.
- Operators: `+ - * / %` (unary, scalar, and component-wise) and comparators (`< > <= >= == !=`,
  returning `boolN`/`boolMxN`) are all **allocating** — each is sugar over a `TempCopy()` plus the
  matching `Comp`/`UnsafeBoolOP` kernel. For a hot loop, call the `*InPlace` methods directly on a
  buffer you own instead (see [comp-elementwise](comp-elementwise.md)).
- `Copy()`/`TempCopy()`/`Dispose()` — only valid on arena-tracked instances (standalone instances
  dispose their own `UnsafeList` directly).

## The two-tier model: authoring vs compute

The API is really two tiers, and the threading contract falls out of which tier you're in:

- **Authoring tier (main thread): the arena.** Operators (`a + b`), `Copy()`/`TempCopy()`, the
  cross-type shortcuts, and the temp pool. Easy arithmetic structurally *requires* the arena: a C#
  operator receives only its operands, so the result's allocator must ride inside them — that is
  what the struct's internal owner reference is for. This tier is for setup, orchestration, and
  gameplay-level math on the scheduling thread.
- **Compute tier (jobs, any thread): pre-allocated buffers + in-place APIs.** `Comp.xxxInPlace`,
  `Blas.dot(..., ref dest)`, solver workspace forms — plus standalone `new floatN(n, Allocator.Temp)`
  scratch, which is thread-local and job-legal. Every kernel in the library itself lives in this
  tier.

The trap to know: **the arena rides invisibly inside every arena-tracked struct**, so an
innocent-looking `var c = a + b;` inside a job is an arena mutation from a worker thread — exactly
the race the contract forbids, reached without ever "passing the arena" anywhere. Inside jobs, use
the in-place APIs on buffers allocated before scheduling; leave the operators to the authoring tier.
(Under `ENABLE_UNITY_COLLECTIONS_CHECKS` the tripwire above turns an actual collision into a thrown
exception; in player builds it is silent corruption.)

## Temp pool & threading in practice

The temp pool is a convenience, not a hard requirement — every allocating op has a `ref`-destination
primitive that writes into a buffer you already own and allocates nothing (see
[zero-alloc-ops](../zero-alloc-ops.md)). Reach for the temp pool in one-shot/setup code and the
zero-alloc primitives inside per-frame loops.

## Benchmarks — arena-sweep record migration, hot-loop cost

The RFC behind this migration (`docs/rfc-memory-model.md`) claimed the record indirection (`Data.Ptr`
resolved once per op, not per element) costs ~0 in a hot loop. Verified by comparing a pre-sweep
worktree (commit `caf0b05`, parent of the first migration commit `e19fc06`) against HEAD (`0714c97`)
on the same machine (AMD Ryzen 9 9950X3D, single CCD pinned), both runs taken back-to-back in the same
session (to control for thermal/frequency drift — see caveat below), Unity Editor batchmode
(`ENABLE_UNITY_COLLECTIONS_CHECKS` likely on; this is not the release/player-build shape), 2026-07-05:

| Kernel | dtype | pre-sweep med(ms) | HEAD med(ms) | delta |
|---|---|---|---|---|
| `matMatDot` (GEMM), N=1024 | float | 23.99 | 24.21 | +0.9% |
| `matMatDot`, N=1024 | double | 46.60 | 44.62 | −4.2% |
| `QR.qrDecomposition`, N=1024 | float | 54.58 | 55.65 | +1.9% |
| `QR.qrDecomposition`, N=1024 | double | 97.35 | 99.17 | +1.9% |
| `BSR.spMV`, N=768, 7% fill | float | 0.3375 | 0.3398 | +0.7% |
| `BSR.spMV`, N=768, 7% fill | double | 0.3443 | 0.3428 | −0.4% |
| `Solvers.cg` (BSR, K=40), N=768, 7% fill | float | 0.0604 | 0.0609 | +0.8% |
| `Solvers.cg` (BSR, K=40), N=768, 7% fill | double | 0.2455 | 0.2461 | +0.2% |

GEMM/QR/BSR-spMV/CG all land within ~2% — consistent with the RFC's "~0 cost" claim. Two kernels
(`LU.decomp`, `CHO.decomp`, float N=1024) showed a larger gap in this same
comparison (+5.3%, +6.6%) that does not reproduce at the same magnitude in double (+0.6%, +3.8%); since
a record-indirection cost should hit both precisions symmetrically (both route through the same
`floatMxN`/`doubleMxN` record type), and since two independent HEAD-only runs of this same suite (see
below) already showed 7–10% self-noise on these exact kernels, this reads as measurement noise rather
than a genuine regression — not confirmed innocent by a tight statistical bound, but not a clear signal
either. A repeat with several runs per side would be needed to shrink the error bars enough to rule
definitively on LU/Cholesky specifically.

**Noise-floor caveat (harness gap found, not fixed):** an *unmatched-session* comparison (pre-sweep run
immediately after a cold Library rebuild, vs. an earlier same-day HEAD run) showed gaps up to 22%
(GEMM) in inconsistent directions — thermal/frequency state after a large import+compile clearly
dominates over any code-level effect. Two back-to-back HEAD-only runs (identical code, ~15 minutes
apart) showed up to 10% self-disagreement on LU. **Take single-run kernel-benchmark numbers on this
harness as ±5–10%, not ±2–3%,** unless runs are taken back-to-back in the same session; a future
improvement would be averaging medians over several process launches, not just several in-process
iterations. Separately, `Tools/benchmark.ps1`'s final "echo the results" step resolves the results
file via a **relative path** parsed from the Unity log, evaluated against the *calling shell's* working
directory — this silently prints a stale/wrong file's contents when the script is invoked against a
project at a different path than the caller's cwd (e.g. a `git worktree`). Not fixed here (benchmark
tooling, out of this task's scope) — read the target project's own `TestResults/benchmark-all.txt`
directly if invoking the script from outside its own project root.
