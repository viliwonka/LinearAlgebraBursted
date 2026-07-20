# Spec: Unity headless test-loop optimisation (task #55)

Read-only investigation of the headless test loop (`Tools/run-tests.ps1`). Goal: cut the
warm (~59s) and cold-recompile (~290s) turnaround **without** weakening test correctness.

## HARD CONSTRAINT (unchanged)

`[BurstCompile(CompileSynchronously = true)]` stays on every test job (165 template files carry
it, e.g. `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/BoolAnalysisTests.cs:12`). It forces
the real Burst path to compile before the test body runs, so a silent Mono fallback can't turn
the suite into a 42-minute crawl or hide a determinism bug. Every recommendation below is
designed to compile **fewer jobs** or **remove non-Burst overhead**, never to defer/skip Burst
compilation of a job that actually executes.

---

## How a run is actually invoked (the facts)

`Tools/run-tests.ps1`:
- Runs `regen.ps1` first unless `-NoRegen` (`run-tests.ps1:39-46`).
- Builds the Unity arg list at `run-tests.ps1:57-65`:
  `-batchmode -projectPath <root> -logFile <log> -runTests -testPlatform EditMode -testResults <xml> -nographics [-testFilter <regex>]`.
  `-nographics` is EditMode-only (`:59`); `-testFilter` is added only when `-Filter` is passed,
  glob `*` -> regex `.*` (`:60-65`).
- `Invoke-Unity` (`_unity-common.ps1:55-91`) starts Unity **without `-Wait`** and blocks on
  `$proc.WaitForExit()` (`:89`) — **no timeout**.

`regen.ps1` (headless, no Unity): prunes orphans (`prune-orphaned-generated.ps1`), `dotnet build`
of `Tools/CodegenBootstrap`, then `dotnet CodegenBootstrap.dll <root>` regenerates the three
Generated trees.

Each `run-tests.ps1` invocation is a **fresh Unity process**: it always pays editor startup +
domain reload(s) + an asset-pipeline refresh. There is no resident editor between runs.

---

## Finding 1 — The cold-recompile trigger: timestamp-churn theory REFUTED; it is content-driven

The theory ("regen rewrites generated `.cs` -> new timestamps -> Burst cache invalidated") is
**wrong**, and the mechanism is worth stating precisely because it changes the fix.

- `CodegenBootstrap/Program.cs:130-137` reads each target file and **skips the write when the
  content is byte-identical** (`if (text == code.text) continue;`). A no-op regen touches **zero**
  generated files — no timestamp change, no reimport. Confirmed this is a line-for-line port of
  Unity's own `ScriptFileGenerator` skip-if-identical behaviour (`Program.cs:98-102`).
- The Burst JIT cache is keyed by an **IL/content hash**, not file timestamps. Even if a file
  *were* rewritten with identical bytes, Roslyn/Bee would emit identical IL -> identical assembly
  hash -> **Burst cache HIT**. A timestamp-only change cannot cause a cold Burst recompile.
- Cache **persistence is confirmed**: `Library/BurstCache/JIT/` holds **8063 `.dll` entries**
  (16128 files incl. `.pdb`) with mtimes spanning **2026-06-12 -> 2026-07-20 coexisting**
  (~68 GB, never pruned). Nothing wipes it between runs.

**Actual trigger of the 290s cold path:** editing a *widely-shared* source file — a core proxy,
`UnsafeOP`, `WideOP`, `Blas`, or a Krylov/block helper that thousands of jobs transitively depend
on — changes those jobs' IL hashes, so they miss the cache and Burst re-JITs them **synchronously**
(CompileSynchronously) the first time each is executed. Editing one leaf kernel only re-JITs its
dependents; editing a shared kernel re-JITs almost everything. This is inherent to the change, not
a spurious invalidation — so the lever is **compile fewer jobs**, not "stop the churn."

Evidence from the in-flight run captured in `TestResults/EditMode.log`: the working tree had many
block-Krylov templates modified (BlockTFQMR / BlockIDR / BlockFGmres / KrylovBlock / KrylovLstsq),
`Asset File Changes: ... changed=3` (`:536`), `CompileScripts: 4602ms` (`:547`), and Burst JIT
warnings (BC1305) streaming *during* test execution (`:623-754`). Even though this run was
**filtered** to `.*DebugInBurst.*`, wall time was ~320s (process 07:46:25Z -> 07:51:45Z) — because
the changed shared code cache-missed the jobs those tests execute.

---

## Finding 2 — The warm floor is Unity startup, NOT test execution or CompileSynchronously

From `TestResults/EditMode.log`, per-process fixed overhead before any test body runs:
- **Three domain reloads**: 1813ms (`:133`) + 2831ms (`:495`) + 3349ms (`:693`) ≈ **8s**.
- **Asset Pipeline Refresh: 7.871s** (`:529`), of which `CompileScripts: 4602ms` (`:547`) — and
  that 4.6s only exists because 3 files changed; a true no-op run compiles ~0.
- First test indexing at ~**10.9s** since startup (`:617`).

So the irreducible warm floor (no code change) is roughly **12–15s of editor startup + domain
reload + refresh**, entirely outside Burst. `CompileSynchronously` does **not** contribute to this
floor. Any "warm" number materially above ~15s is either the full 6538-test execution sweep or
Burst re-JIT of changed jobs — not startup.

---

## Finding 3 — `-Filter` already exists and is the main dev-loop lever

`run-tests.ps1` already accepts `-Filter` (glob or regex; `:60-65`), and the in-flight run proves
it's wired to Unity's `-testFilter` (`EditMode.log:597 groupNames = .*DebugInBurst.*`).

Because Burst compiles **lazily, per executed job**, a filtered run only JIT-compiles the jobs the
selected tests actually run. Full-suite cold = every job; single-fixture cold = a handful. This
**fully preserves CompileSynchronously** (the jobs that do run still compile synchronously before
their test body). Caveat from Finding 1: if you edited *shared* code, even a small fixture's jobs
cache-miss, so the filter caps the count but each still pays a synchronous compile.

---

## Finding 4 — The "hang": two distinct causes, both fixable

1. **Coordination, not Unity.** `TestResults/agent-run.log` contains exactly `pwsh: command not
   found`. An agent launched a background run with `pwsh` (PowerShell Core, not installed — this
   box is Windows PowerShell 5.1 / `powershell.exe`). The background launch died immediately while
   the launching agent parked waiting for output that never came. This reads as a "hang" but is a
   launcher bug.
2. **No wall-clock guard.** `Invoke-Unity` blocks on `$proc.WaitForExit()` with no timeout
   (`_unity-common.ps1:89`). A genuinely stalled editor (license wait, a deadlocked job, a Burst
   stall) blocks the script — and any agent awaiting it — forever. The test-runner's own
   `playerHeartbeatTimeout = 600` (`EditMode.log:591`) is PlayMode-only and does not cover this.

---

# Ranked recommendations

Ordering = (impact on the painful cold path) x (safety). All preserve CompileSynchronously.

### R1 — Inner-loop = `-Filter` to the touched fixture (biggest cold-path win) — DOC/HABIT, no code
- **Change:** For iteration, run `./Tools/run-tests.ps1 -Filter "*<Fixture>*"`; reserve the full
  suite for pre-commit. Document this front-and-centre (it already works; the gap is habit).
- **Saving:** Cold path drops from "re-JIT every job" to "re-JIT only the selected fixture's
  jobs" — from ~290s toward tens of seconds when the change is localised. Warm filtered runs land
  near the ~12–15s startup floor.
- **Risk:** Low. Only risk is *forgetting* to run the full suite before commit — so gate commits
  on the unfiltered run.
- **CompileSynchronously:** Preserved — selected jobs compile synchronously; unselected jobs
  simply never execute.

### R2 — Add `-NoRegen` to the habit when you didn't touch a template — DOC/HABIT, no code
- **Change:** `run-tests.ps1 -NoRegen` (flag already exists, `:19-21,:39`) when re-running the same
  code, or when iterating on non-templated harness code you regenerated once already.
- **Saving:** Removes the `dotnet build` of CodegenBootstrap + the full prune scan of all three
  Generated trees each run (seconds, and it serialises before Unity even launches).
- **Risk:** Low, with one footgun: if you *did* edit a template and pass `-NoRegen`, you test stale
  generated code. Keep regen ON for the pre-commit full run.
- **CompileSynchronously:** Unaffected.

### R3 — Hang-proof `Invoke-Unity` with a hard wall-clock timeout — SCRIPT CHANGE
- **Change:** Add a `-TimeoutSec` param (suggest default 900s; a healthy full cold run is <300s) to
  `Invoke-Unity` (`_unity-common.ps1:55-91`). Replace the bare `$proc.WaitForExit()` (`:89`) with
  `$proc.WaitForExit($TimeoutSec*1000)`; on timeout, kill the process tree
  (`Stop-Process -Id $proc.Id -Force`, plus any child Unity/bee), write a clear
  `FAIL: Unity exceeded <n>s wall clock — killed` line, and return a non-zero code so callers/agents
  fail fast instead of parking. Thread the param through `run-tests.ps1`/`benchmark.ps1`.
- **Saving:** Converts an unbounded hang into a bounded, reported failure. No steady-state speedup,
  but removes the worst-case (infinite park).
- **Risk:** Low. Pick the timeout comfortably above the real cold-full-suite time so a legitimately
  slow cold run isn't killed. Killing mid-Burst-compile can leave a partial cache entry, but the
  cache is content-hash-keyed so a partial/renamed temp is simply re-created next run (Burst writes
  atomically to the final hash name).
- **CompileSynchronously:** Unaffected.

### R4 — Fix background launches to use `powershell.exe`, not `pwsh` — HARNESS/DOC
- **Change:** Any agent/wrapper that launches the suite in the background must call
  `powershell.exe -NoProfile -File Tools/run-tests.ps1 ...` (not `pwsh`). `pwsh` is not installed
  (`agent-run.log`), so the background job dies instantly and the caller parks. Also prefer running
  the suite in the **foreground** (it holds the single Unity build slot anyway — background buys
  nothing and invites the park).
- **Saving:** Eliminates the false "hang" class entirely.
- **Risk:** None.
- **CompileSynchronously:** Unaffected.

### R5 — (Optional, nice-to-have) `-Changed` auto-filter mode — SCRIPT CHANGE
- **Change:** Add a mode to `run-tests.ps1` that reads `git status --porcelain` for touched
  `TemplateSourceTests/**/<Name>*.cs` and builds a `-testFilter` from the fixture name(s)
  automatically, so the inner loop needs no manual `-Filter`. Fall back to the full suite when the
  change touches shared *source* (non-test) templates (those can invalidate anything, so a full run
  is the honest choice).
- **Saving:** Makes R1 automatic; same ceiling as R1.
- **Risk:** Medium — the shared-source fallback heuristic must be conservative (when unsure, run
  everything) or it silently under-tests. Land only with that guard.
- **CompileSynchronously:** Preserved.

### R6 — (Optional, housekeeping) Bounded `BurstCache/JIT` pruning — MANUAL ONLY, do NOT automate
- **Observation:** 68 GB / 8063 unpruned JIT entries. This does **not** slow per-run lookups
  (hash-indexed) and is **not** a correctness issue — it's disk bloat only.
- **Change:** If disk pressure bites, delete `Library/BurstCache/JIT` manually during downtime.
- **Risk:** **The next run after a clear is fully cold (~290s+) — every executed job re-JITs.**
  Never wire this into `run-tests.ps1` and never clear it as part of the loop; that would *cause*
  the cold path it's meant to avoid. Also never `git clean`/wipe `Library/` in the loop for the
  same reason.
- **CompileSynchronously:** Unaffected.

### Not recommended / dead ends
- **Removing/relaxing CompileSynchronously** — forbidden (hard constraint; hides Mono fallback +
  determinism bugs).
- **Chasing "codegen timestamp churn"** — refuted (Finding 1); no such churn exists on a no-op
  regen. Don't add "skip regen to protect the Burst cache" logic — it protects nothing.
- **Trying to eliminate the ~12–15s startup floor** — it's per-process editor startup + domain
  reload; there is no resident-editor headless test server in this setup. Out of scope. (Keeping
  the GUI editor closed and the scripting-define/AOT settings stable between runs avoids *extra*
  reloads, which is already the case.)

---

## One-line summary
The cold 290s is real Burst re-JIT of shared-code-dependent jobs (not timestamp churn — the cache
persists and is content-hashed); the warm ~59s is dominated by a ~12–15s editor-startup floor plus
the full-suite execution sweep. Fastest safe wins, in order: **filter to the touched fixture (R1)**
+ `-NoRegen` (R2) for the inner loop, a **wall-clock kill timeout (R3)** and **`powershell.exe` not
`pwsh` (R4)** to make the loop hang-proof — all of which keep `CompileSynchronously` intact.
