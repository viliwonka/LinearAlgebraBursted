# Spec: universal benchmark + hash harness (player-buildable scene)

Status: ready for implementation (coder agent).
Origin: user request 2026-07-17. Internal doc — `docs/dev/` is exempt from the public-docs prose bar.

Companion spec: `docs/dev/spec-determinism-conformance-harness.md` ("the determinism spec" below).
That spec owns the hash-group taxonomy (sections A/B, ~25+6 groups), the op/group/root fold
definitions, the report line grammar, and the binding determinism rules (§5 there). **This spec does
not restate any of that — it references it.** This spec is the determinism spec's "player-build
runner" open question (its §11) made concrete, plus a benchmark-timing phase.

## 1. Goal

One Unity SCENE, buildable into a standalone player, that on launch:

1. runs every determinism hash group (section A + B of the determinism spec) and prints/writes the
   per-op, per-group, ROOT and ROOT-B hashes, and
2. runs the full benchmark suite (the same sections `AllBenchmarks.Run` composes) and prints/writes
   the timing tables,

writing both to files under `Application.persistentDataPath` and mirroring them on screen. The same
binary (per platform) runs on x86 AVX/AVX2 machines, ARM NEON machines, and — via a Burst-AOT-off
build — the Mono/IL fallback path. Workflow: run the binary on each machine, collect the hash
files, `diff` them (section-A divergence = bug; section-B divergence = expected only under
`LINALG_NATIVE_MATH`), and compare the timing files by eye.

Why a player and not the Editor: the Editor only exists on the dev box's arch. Cross-arch and
Burst-vs-Mono coverage needs a shippable executable. The existing `Tools/benchmark.ps1` /
`Tools/determinism-report.ps1` (per the determinism spec) remain the Editor-side siblings; this is
the same code reached from a `MonoBehaviour` instead of `-executeMethod`.

## 2. Non-goals

- No new hash groups, no new taxonomy — groups are the determinism spec's, verbatim, same
  `HarnessRev` gating.
- No new benchmark kernels — reuse the existing `*Benchmark.Section(sb)` bodies unchanged.
- Not shipped: nothing under the UPM package root (`Assets/LinearAlgebra/Source`) changes. The
  harness is a dev tool in the project, like the demos.
- No networked result collection, no result database — files + `diff`.
- No UTF/NUnit anywhere: players have no Unity Test Framework. Plain `MonoBehaviour` + Burst jobs.
- v1 targets standalone desktop (Windows/macOS/Linux, x64 + arm64) and optionally Android; no
  iOS/console packaging work in v1 (nothing in the design blocks it later).

## 3. Prerequisite + assembly plumbing (the one structural decision)

The determinism spec places its generated case runners in `TemplateSourceBenchmarks` →
`Benchmarks/Generated`, i.e. inside `BurstLinearAlgebra.Benchmarks` — which is currently
`"includePlatforms": ["Editor"]`. A player scene cannot reference an Editor-only assembly.

**DECISION (user, 2026-07-17): do NOT flip `BurstLinearAlgebra.Benchmarks` to all-platforms.** It
stays Editor-only. The player harness reaches the reusable bodies through a separate player-visible
assembly — but implemented in the drift-free form, NOT as re-declared shim copies:

- **Relocate the player-clean bodies down into a new all-platforms assembly**
  `BurstLinearAlgebra.BenchCore` (`"includePlatforms": []`): the `Bench` helper, the
  `*Benchmark.Section(sb)` bodies, `AllBenchmarks.Sections(...)`, and the determinism spec's
  `DeterminismReport.Build(sb)` + its generated case runners. These are player-clean by
  construction (plain statics, Burst IJobs, `System.*`/`Unity.Burst`/`UnityEngine.Debug`, no NUnit,
  no `-executeMethod` baked in).
- **`BurstLinearAlgebra.Benchmarks` keeps only the Editor-side entry points** and references
  `BenchCore`: `[MenuItem]` wrappers, the `-executeMethod` `Run()`/`DeterminismReport.Run()`
  wrappers that write `TestResults/…`, and anything `UnityEditor`-touching (wrapped `#if
  UNITY_EDITOR`). It re-exports nothing — it just calls into `BenchCore`.
- **Both the Editor Benchmarks assembly and the player harness reference `BenchCore`.** ONE copy of
  every section/case body, one `Sections(...)` list — so there is no shim duplication and nothing to
  drift (the risk that killed the plain-shim option). This is the "separate assembly" the user
  chose, minus the copy-paste.
- Codegen note: point `TemplateSourceBenchmarks`/determinism codegen output at `BenchCore`'s folder
  (or add `BenchCore` as the target and leave the Editor asmdef body-free). Coder verifies
  `grep -r "UnityEditor" <BenchCore source>` is empty before wiring.
- Cost vs the flip: only `BenchCore` (not the Editor menu/report plumbing) compiles into player
  builds of this project. Zero effect on the UPM package (`Source/` only).

Prerequisite ordering: the determinism harness (its spec) may or may not be implemented yet.

- If NOT yet implemented: implement it FIRST per its own spec, with the amendments below applied
  from the start.
- If already implemented: apply the amendments as a small refactor.

Amendments to the determinism-harness implementation (record in
`TemplateSourceBenchmarks/DEVLOG.md`, and treat this section as overriding the determinism spec
where they touch):

- **A1** — extract the player-clean bodies into the all-platforms `BurstLinearAlgebra.BenchCore`
  assembly (§3); `BurstLinearAlgebra.Benchmarks` stays Editor-only and references `BenchCore`.
- **A2** — split the entry point: `DeterminismReport.Build(StringBuilder sb)` (pure: runs all
  groups, appends the report body — `rev`, `#` header lines, `ROOT`, `ROOT-B`, `GROUP`/`OP` lines —
  returns nothing else; NO file I/O, NO `UnityEditor`, no `Debug.Log`) and
  `DeterminismReport.Run()` (Editor `-executeMethod` wrapper: calls `Build`, writes
  `TestResults/determinism-report.txt`, logs the path — exactly the contract its spec §7/§8 fixes).
  The player harness calls `Build` and does its own writing.
- **A3** — the `#` header lines come from a shared `PlatformId` helper (§6 below) so Editor
  reports and player reports have identical header vocabulary.
- **A4** — `Bench.Warmup`/`Bench.Runs` become `public static int` fields (same defaults, 1 and 4)
  instead of consts, so the player harness can raise them without touching the EditMode wrapper's
  behavior. `Bench.WriteReport` is untouched.

## 4. File/dir layout

```
Assets/ConformanceHarness/
  ConformanceHarness.unity          scene: one camera, one GameObject with HarnessMain
  ConformanceHarness.asmdef         all-platforms; refs: BurstLinearAlgebra,
                                    BurstLinearAlgebra.BenchCore, Unity.Burst,
                                    Unity.Collections, Unity.Mathematics; allowUnsafeCode
  HarnessMain.cs                    MonoBehaviour driver + OnGUI screen mirror
  PlatformId.cs                     SIMD-caps probe job + Burst-vs-Mono probe + header/tag builder
  HarnessWriter.cs                  persistentDataPath writer (UTF-8 no BOM, '\n')
  Editor/
    ConformanceHarness.Editor.asmdef  Editor-only
    BuildHarness.cs                 menu/batch build methods (§9)
```

All hand-written (no codegen: the generated work already lives in `Benchmarks/Generated`). Nothing
under `Assets/LinearAlgebra/Source` — the UPM package is untouched. `Assets/Scenes/` holds only
`SampleScene.unity` and `Assets/Demos/` is the numbered demo area; a dedicated top-level
`Assets/ConformanceHarness/` keeps the harness out of the demo list (it is not a demo).

## 5. Runtime flow (HarnessMain)

`Start()` launches a coroutine; each step yields between groups/sections so the screen repaints and
the OS doesn't flag the app as hung (single frames may still block for seconds during a heavy
benchmark size — acceptable, this is a lab tool).

```
Awake:  Application.runInBackground = true; Screen.sleepTimeout = SleepTimeout.NeverSleep;
        QualitySettings.vSyncCount = 0; Application.targetFrameRate = -1;
Phase 0 — identify:   PlatformId probe (§6); build header + platform tag; show on screen.
Phase 1 — hashes:     DeterminismReport.Build(sb) with per-group yield (§5a);
                      write hashes-<tag>.txt; show ROOT/ROOT-B on screen.
Phase 2 — timings:    Bench.Warmup = cfg.warmup; Bench.Runs = cfg.runs;
                      foreach section in the AllBenchmarks order: Section(sb); yield;
                      write bench-<tag>.txt.
Done:   show both absolute file paths on screen (large font — the user reads them off the device).
```

- **§5a per-group yield**: `Build` appending everything in one call blocks the screen for the whole
  hash phase. Either (i) `Build` stays monolithic and phase 1 is one long hitch (~low minutes,
  determinism-spec budget) with a "hashing… (screen will freeze)" notice — acceptable for v1 — or
  (ii) `DeterminismReport` additionally exposes its group list as
  `IReadOnlyList<(string name, Action<StringBuilder, RootAccum>)>` so the coroutine can iterate
  with progress. Coder's choice; (ii) is worth it if cheap, but do not let it perturb the report
  byte layout (acceptance: player hash body == Editor hash body, §10.3).
- **Config** (quick vs full), read in this priority: command-line `--quick` / `--hash-only` /
  `--runs=N` (`Environment.GetCommandLineArgs`; fixed config inputs, not nondeterminism), else two
  on-screen buttons ("Full run", "Quick run") before phase 1 on platforms without a command line
  (Android). Quick run = hash phase in full (it is the cheap, high-value part) + benchmarks
  restricted to sizes ≤ 256. **Auto-quick**: if the Burst probe (§6) reports Mono fallback, default
  to quick — the full suite under Mono is the 42-minute-suite trap; the header records
  `mode: quick(auto-mono)`.
- Benchmark ordering is exactly `AllBenchmarks.Run`'s section order (stable across platforms so
  the timing files line up row-for-row for eyeball or script comparison). Do not call
  `AllBenchmarks.Run` itself (it writes to `TestResults/` via `Bench.WriteReport`); call the
  `Section(sb)` methods, same list, and keep the list in ONE place — add
  `AllBenchmarks.Sections(StringBuilder sb, Action perSectionYield = null)` and make `Run()` use
  it, so the Editor path and the player path cannot drift.

Timing methodology (noisy-machine requirement): unchanged mechanism — `Bench.Time` already does
warmup then N timed runs and reports the **median** (`Bench.Row` computes GFLOP/s from the median;
time-only rows for iterative kernels). The harness raises the knobs: player defaults
`Warmup = 2`, `Runs = 9` (median of 9 tolerates ≥4 outlier samples; full run stays well under an
hour under Burst at current sizes). `--runs=N` overrides. Optional (Windows standalone only,
post-MVP): set `Process.GetCurrentProcess().ProcessorAffinity` from `--affinity=0x<mask>` to mirror
`benchmark.ps1`'s CCD pinning; default no pinning.

## 6. PlatformId: self-identifying header + filename tag

Two probes, because managed-side facts and Burst-side facts differ:

1. **Burst-vs-Mono probe** (the standard `[BurstDiscard]` trick): a `[BurstCompile]` IJob whose
   `Execute` sets `flag[0] = 1` and then calls a `[BurstDiscard]` method that sets `flag[0] = 0`.
   After `.Run()`: 1 ⇒ the job ran Burst-compiled native code; 0 ⇒ Mono/IL fallback. This is the
   truth for the actual kernels, unlike any editor-only `BurstCompiler.Options` query.
2. **SIMD-caps probe**, inside the same Burst job (the `X86.Avx.IsAvxSupported` /
   `X86.Avx2.IsAvx2Supported` / `X86.Fma.IsFmaSupported` / `Arm.Neon.IsNeonSupported` properties
   are compile-time-folded by Burst; evaluated from managed code they are `false`): write each as
   0/1 into a `NativeArray<int>`. Under Mono fallback they all report 0 — which is accurate: the
   fallback scalar path is what executes (`WideOP` takes its `math.*` lanes).

Header lines (same `#` grammar as the determinism spec §7; informational, never hashed, identical
vocabulary in Editor and player reports — amendment A3):

```
# host: <SystemInfo.operatingSystem> / <SystemInfo.processorType> / <SystemInfo.processorCount>t
# unity: <Application.unityVersion>  backend: <Mono|IL2CPP>  config: <debug|release>
# burst: <True|False>  simd: <avx2+fma|avx|neon|scalar>
# detmath-native: <DetMath.UseNative>
# harness: rev <HarnessRev>  mode: <full|quick|hash-only>  warmup <W> runs <R>
```

Backend via `#if ENABLE_IL2CPP`; config via `Debug.isDebugBuild`. No timestamps anywhere in the
hash file (determinism spec §5.7 binds); the timing file MAY carry a date line — it is never
diffed for identity.

Filename tag (stable per machine+build, so re-runs overwrite — intended):
`<os>-<arch>-<simd>-<burst|mono>`, e.g. `hashes-win-x64-avx2+fma-burst.txt`,
`bench-macos-arm64-neon-burst.txt`, `hashes-win-x64-scalar-mono.txt`. Lowercase, no spaces.

## 7. Output files

Two files, not one — the hash file must stay byte-diffable, timings are inherently per-machine:

- **`hashes-<tag>.txt`** — exactly the determinism spec's report format (§7 there): `rev` line, `#`
  header lines, `ROOT`, `ROOT-B`, `GROUP`/`OP` lines, section-B fence. The BODY (everything except
  `#` lines) must be byte-identical to what the Editor wrapper produces on the same
  machine/build-mode — same `Build()` code path guarantees it. Cross-machine workflow: the
  determinism spec's `Tools/determinism-report.ps1 -Compare a b` works on these files unmodified
  (it ignores `#` lines and gates on `rev`).
- **`bench-<tag>.txt`** — the same `#` header block, then the concatenated `Section(sb)` tables
  (existing `Bench.Header/Row` format: dtype, N, min/med/mean/max ms, GFLOP/s from median).

Sample (hash file, abbreviated — the format authority is the determinism spec):

```
=== LinearAlgebra determinism conformance report ===
rev 1
# host: Windows 11 (10.0.26200) / AMD Ryzen 9 9950X3D / 32t
# unity: 6000.0.32f1  backend: Mono  config: release
# burst: True  simd: avx2+fma
# detmath-native: False
# harness: rev 1  mode: full  warmup 2 runs 9

ROOT 4f1d9c2a
ROOT-B 913bb0e7

GROUP blas-dense 5a77d001
OP blas-dense/dot.vv.f32.n1000 c0ffee12
...
```

Writing: `HarnessWriter` writes UTF-8 **no BOM**, `'\n'` line endings (determinism spec
convention), via `File.WriteAllBytes(Path.Combine(Application.persistentDataPath, name), …)`. Also
`Debug.Log` both full paths, and keep the last ~40 report lines in the OnGUI mirror with the two
paths pinned at the top.

## 8. Hashing + determinism specifics (what this spec adds on top of the determinism spec)

- Hash algorithm/API: unchanged — `LinearAlgebra.Hash` xxHash32:
  `Hash.hash(byte* data, int byteLength, uint seed = 0)` (core), typed wrappers
  `Hash.hash(in fProxyN)` / `Hash.hash(in fProxyMxN)` / int-family, `Hash.combine(uint, uint)` as
  the fold. All hashing stays inside the Burst jobs per the determinism spec §3.
- Fixed-seed inputs: bound by determinism spec §5.1 (literal `Unity.Mathematics.Random` seeds,
  never `DateTime`/`Environment.TickCount`/wall clock). The harness adds NO new input sources; the
  command-line/config values do not feed any hashed computation.
- Both dtypes: the generated float + double halves both run — bound by the determinism spec (§6:
  every group runs float AND double except where noted).

**NaN / native max-min semantics (the trap, spelled out).** `FloatMode.Strict` makes `+ - * /
sqrt` cross-arch bit-identical, and DetMath routes transcendentals deterministically — but
min/max is a third category: `WideOP.fProxy.cs` `Max`/`Min` use `X86.Avx.mm256_max_ps/pd`
(`mm256_min_ps/pd`) on AVX and per-lane `math.max`/`math.min` otherwise. For NaN operands these
disagree: the AVX instruction returns the SECOND operand whenever either input is NaN (so NaN
propagates or is suppressed depending on operand order), while `math.max` has its own asymmetric
NaN rule — and mixed-sign zero ties (`max(-0,+0)`) are likewise operand-order-dependent on AVX.
Same op, same FloatMode, legitimately different bits AVX-vs-fallback.

Resolution (both halves, and this is the contract):

1. **Section A inputs are NaN-free and signed-zero-tie-free by construction.** The determinism
   spec's recipes are seeded uniform fills / gallery matrices — no NaNs, and
   `Random.NextFloat`-style generation never produces `-0`. The coder must PRESERVE this property
   in every case that feeds a min/max/argmin/argmax/clamp/saturate kernel (groups
   `elementwise-core`, `stats-core`, `norms` LInf, `histogram-resample-query`): no NaN injection,
   no deliberate ±0 pairs. With finite, tie-free-in-sign inputs, `mm256_max_ps` and `math.max`
   agree bit-exactly, so these groups stay in section A legitimately.
2. **A `nan-minmax` section-B group IS in scope (user, 2026-07-17).** It deliberately feeds NaN
   operands and mixed-sign-zero ties (`max(-0,+0)`, both operand orders) through the min/max/HMax/
   HMin kernels so the AVX-vs-scalar divergence is MEASURED, not merely avoided. Folds into ROOT-B
   only, per the determinism spec's section-B machinery; documented in the report fence as
   expected-to-differ AVX-vs-scalar (and Burst-vs-Mono). Keep it strictly out of ROOT/section A so
   section-A stays cross-platform-stable. Covers both float (8-lane, `mm256_max_ps` vs the new
   `float4x2` two-halves fallback) and double (`mm256_max_pd` vs `math.max`).

**Burst-off runs are in-scope for hashing.** A Mono-fallback binary produces a full hash file; the
header says `burst: False`. Diffing it against a Burst file checks IL-vs-native agreement — that
diff is *informational*, not a contract (the library's determinism contract is stated for
Burst/Strict). The compare script treats it like any other diff; the human reads the headers.

## 9. Build path (Editor/BuildHarness.cs)

Menu items + batch-callable statics (`-executeMethod ConformanceHarness.Editor.BuildHarness.<M>`):

- `BuildWindowsX64()`, `BuildMacArm64()`, `BuildLinuxX64()`, `BuildAndroidArm64()` — release,
  IL2CPP where available (else Mono), Burst AOT ON, scene list = `ConformanceHarness.unity` only,
  output `Builds/Conformance/<target>/`.
- `BuildWindowsX64MonoFallback()` — Mono backend, **Burst AOT disabled** (set
  `BurstPlatformAotSettings` `EnableBurstCompilation = false` for the target during the build,
  restore after; this is the supported per-platform AOT switch). This binary exercises the
  managed fallback path end-to-end.
- Keep each method dumb: set settings, `BuildPipeline.BuildPlayer`, restore settings, log output
  path. No build-matrix framework.
- Optional post-MVP: `Tools/build-conformance.ps1` wrapping these via `Invoke-Unity`, modeled on
  `Tools/benchmark.ps1` (dot-source `_unity-common.ps1`, run `regen.ps1` first, fail on
  `Get-CompileErrors`).

Burst AOT note: desktop x64 players should build with the default targets-all
(SSE2+AVX2 multi-target) so one Windows binary exercises AVX2 on an AVX2 machine and SSE2 on an
older one — the caps probe (§6) reports which path actually ran, which is exactly what the header
is for. Do not pin `OptimizeFor`/target CPUs narrower without recording it in the header.

## 10. Acceptance criteria

1. Project compiles for Editor AND for a Windows x64 player build; full test suite still green
   (`Tools/run-tests.ps1`); `Tools/regen.ps1 -Check` clean.
2. Editor determinism wrapper (`Tools/determinism-report.ps1`) still works and its report body is
   unchanged by the refactor (amendments A2–A4 are behavior-preserving in the Editor path).
3. **Editor/player hash parity**: on the dev machine, the player build's `hashes-<tag>.txt` body
   (all non-`#` lines) is byte-identical to the Editor-produced report body. This is the proof
   that the scene runs the same code path.
4. **Re-run identity**: running the same player binary twice yields byte-identical hash files.
5. Mono-fallback build (Burst AOT off) runs to completion in quick mode, writes both files,
   header says `burst: False`, `simd: scalar`.
6. Bench file contains every section `AllBenchmarks.Run` contains, in the same order, with
   median-based rows; `warmup`/`runs` header values reflect the config actually used.
7. On-screen output shows: header, progress, ROOT/ROOT-B, and both absolute output paths.
8. Nothing added/changed under `Assets/LinearAlgebra/Source`; comment policy respected (rationale
   → `TemplateSourceBenchmarks/DEVLOG.md` and/or a new `Assets/ConformanceHarness/DEVLOG.md`).

## 11. Phasing

- **Phase 0 (prerequisite)**: determinism harness per its own spec + amendments A1–A4. (Skip the
  parts already implemented if any; apply amendments.)
- **Phase 1 (MVP)**: scene + asmdef + HarnessMain + PlatformId + HarnessWriter; hash phase only
  (`--hash-only` default OFF but the phase-2 code may be stubbed); Windows x64 Burst build +
  Mono-fallback build; acceptance 1–5, 7, 8.
- **Phase 2**: benchmark phase (Sections reuse, knob raising, quick/full modes); acceptance 6.
- **Phase 3 (optional)**: build script wrapper, `--affinity` pinning, Android build method polish,
  default-mode sentinel groups (below) if approved.

## 12. Open questions for the user

RESOLVED 2026-07-17: **(1) asmdef** — do NOT flip Benchmarks; extract player-clean bodies into
`BenchCore` (§3, no shim duplication). **(3) Mono default** — quick mode (full hashes + benches
capped ≤256), auto-selected under Mono fallback. **(5) nan-minmax** — YES, add the section-B group
(§8.2), divergence measured.

Still open:

2. **Player timing knobs**: Warmup=2 / Runs=9 median — enough for the noisy-PC concern, or do you
   want more (e.g. Runs=15) at the cost of run length? Also: should the EditMode
   `benchmark.ps1` path adopt the raised defaults too, or stay at 1/4?
4. **FloatMode.Default sentinel groups** (determinism spec §11 first bullet): this player harness
   is where Default-mode divergence would actually matter (game builds run Default). Add 2–3
   sentinel groups under a separate `ROOT-C` fence in a later phase? (Cross-arch divergence there
   would be expected-possible, like section B.)
6. **Android**: is an Android arm64 build wanted in v1 (NEON coverage without ARM desktop
   hardware), or is a macOS/Windows-on-ARM machine the intended ARM target?
7. **Output naming**: overwrite-per-tag filenames (proposed) vs. keeping N previous runs — any
   need for run history on-device?
