# Spec: test/benchmark Burst-vs-Mono hygiene, and Krylov test naming/battery integration

Status: survey + spec, not implemented. Read-only investigation; no code changed.

## 0. Problem statement

Most test files under `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/*.cs` put NUnit
`Assert.*` calls **inside** the `[BurstCompile] struct … : IJob { void Execute() }` body, with the
`[Test]` method reduced to a bare `.Run()`. Two costs:

1. **Speed.** A construct Burst cannot translate makes the whole job fall back to interpreted
   execution — 25-100x slower — and this project has already been bitten by it once at suite scale
   (see §1.3).
2. **Coverage (the important one).** A Mono-executed job never runs the Burst-compiled machine code,
   so it cannot catch Burst-specific bugs (`IJob` struct-copy-on-schedule, uninitialized/`0xNaN`
   Temp memory, aliasing). Separately — and this is the larger share of what the 2026-07-20 Fable
   Krylov audit (`docs/dev/audit-krylov-fable-20260720.md`) actually found — batteries can also miss
   bugs simply because **no test exercises the scenario at all** (e.g. no warm-started solve was ever
   tested), independent of Mono/Burst. §6 addresses that half by folding scenario coverage into the
   battery instead of one-off files.

This spec also covers the parallel benchmark-hygiene concern (§5) and the test naming/battery
integration ask (§6), triggered by `KrylovAuditRegressionTests.fProxy.cs` — a new, uncommitted file
(`git status` shows it untracked) that the project owner has already flagged as unacceptably named
and structurally isolated from the existing battery.

## 1. Diagnosis

### 1.1 How widespread the assert-inside-`Execute` pattern is

Scan of `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/*.cs` (149 files):

| Bucket | Count | Meaning |
|---|---|---|
| Contain an `IJob` struct | 139 / 149 | |
| **100% of `Assert.*` calls are inside the `IJob` struct** | 43 | worst case — `[Test]` is a bare `.Run()` |
| Mixed: some `Assert.*` inside the struct, some in the managed `[Test]`/wrapper | 89 | mostly the "battery" idiom (§1.2) |
| **Zero** `Assert.*` calls inside any `IJob` struct | 7 | the clean pattern, or deliberately no-job (§1.2/1.4) |

3,126 total `Assert.*` calls across these files; 147/149 files call `Assert.*` somewhere. Put
together, **132/139 (95%) of IJob-based test files have at least one `Assert.*` call inside
`Execute()`**. This is the concrete basis for "almost every unit test."

Representative **bad**-pattern file: `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ConjugateGradientTests.fProxy.cs`.
The struct is declared at line 12 (`public struct ConjugateGradientTestJob : IJob`); every
`Assert.IsTrue`/`Assert.IsFalse` call in the file (lines 119, 144, 165, 168, 186, 195, 197, 216,
221-222, 232-233, 252-253, 271, 296, 301, 321, 326, 329, 332, 349, 352, 372, 375, 394, 397) is
inside `Execute()`'s call graph. Every `[Test]` method is a one-liner, e.g. lines 403-407:

```csharp
[Test]
public void AddScaledInPlaceTest()
{
    new ConjugateGradientTestJob() { Type = ConjugateGradientTestJob.TestType.AddScaledInPlace }.Run();
}
```

### 1.2 The nuance: not every `Assert.*` inside `Execute` silently defeats Burst

Burst ships a curated intrinsic allowlist for a subset of NUnit `Assert` overloads specifically so
test authors can assert inside jobs. This project has already hit both edges of that allowlist and
documented them in-repo:

- `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/LQRPTests.fProxy.cs:1208-1211` (own
  project comment, not a memory): *"Records an early-out failure into the Fail[] array ONLY — no
  in-Burst Assert.Fail(string): that overload is not Burst-compilable (BC1071) and, worse, silently
  drops the whole job to a Mono fallback."*
- `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KalmanTests.fProxy.cs:22`: *"route every
  numeric assertion through Fail[0..3] with IsTrue-style checks (BC1330 forbids enum
  Assert.AreEqual inside Burst)."*
- `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovAuditRegressionTests.fProxy.cs:33`:
  *"Assertions carry NO message string (the NUnit params-object overload is unsupported under Burst
  -- BC1071)."*

So the concretely unsafe constructs are: `Assert.*(…, string message)` overloads (BC1071),
`Assert.AreEqual`/`AreNotEqual` with enum arguments (BC1330), and by extension anything requiring
managed formatting (string interpolation, `ToString()` on non-primitives, boxing). The safe,
apparently-Burst-compiled subset is the bare-bool overloads: `Assert.IsTrue(bool)`,
`Assert.IsFalse(bool)`.

**Direct runtime proof that the bare-bool overload really does execute as Burst-compiled native
code**, not Mono: `TestResults/EditMode.log:773-786` (a stored run of exactly
`MinresQLPWarmStartRecoveredTest`, i.e. `fProxyKrylovAuditRegressionTests.AuditRegressionJob.MinresQLPWarmStartRecovered`)
shows a `burst_abort` native trap, not a Mono JIT frame, at the assert site:

```
errorId: Assert failed
This Exception was thrown from a job compiled with Burst, which has limited exception support.
0x00007ff9b7a6dc0b (Unity) burst_abort
0x00007ff99faeaede (...) burst_Abort_Trampoline
0x00007ff99f7d57ce (...) fProxyKrylovAuditRegressionTests.AuditRegressionJob.MinresQLPWarmStartRecovered (at .../KrylovAuditRegressionTests.fProxy.cs:0)
```

(Everything below that frame — `IJobExtensions:Run`, `MinresQLPWarmStartRecoveredTest` — is
`(Mono JIT Code)`, which is normal: that's the *managed test harness* calling into the job, not the
job body itself.)

**Implication for this spec**: the risk is real but the mechanism is narrower than "every
`Assert.*` call forces Mono." The actual risks worth fixing, in order of severity:

1. **Fragility.** The safe/unsafe boundary (bare bool vs. anything else) is unwritten, easy to cross
   accidentally (adding a message string "for debuggability" is the natural next edit), and when
   crossed the failure mode is the worst kind — see 1.3.
2. **Diagnostics.** `Assert.IsTrue(ok)` with no message gives literally `"Assert failed"` and nothing
   else (see the log above) — the project's whole `Fail[]`-array convention (§1.4) exists to work
   around this, by recording context that gets formatted into a message *outside* the job.
3. **Hard-abort-on-first-failure.** Burst's "limited exception support" (its own log message) means
   an assert failure inside `Execute()` aborts the entire job immediately. In a battery loop
   (`RunStandardChecks` iterating many gallery matrices), this means only the *first* failure is ever
   observed in a run — you cannot see whether one matrix or all of them are broken. (The existing
   `Fail[0] == 0` guard in `Record()` already only captures the first failure anyway — see §1.4 — so
   this is not currently losing information, but it forecloses ever *improving* that.)
4. **Unverified claim about compile-time errors.** Whether a genuine BC compile-time diagnostic
   (BC1330/BC1071) under `CompileSynchronously = true` reliably *throws* at `Run()` (per the
   `faster-testing-task`/`burst-test-compile-gotchas` project memory, dated 2026-07-06) versus
   *silently* falls back is asserted but not something this read-only survey re-verified by running
   Unity. §2 gives the implementer a way to check this directly rather than trust the memory.

### 1.3 Why this matters at suite scale — prior incident

Already happened once, documented in this project's own memory (`burst-test-compile-gotchas`,
`faster-testing-task`, 2026-07-06): a single BC1330 (enum `Assert.AreEqual` inside a job) in the SVD
solver battery silently dropped that whole `TestJob` to Mono, at 25-100x slowdown, contributing to a
10+ minute suite (now ~49s). The mitigation shipped then was `CompileSynchronously = true` on all
367 `[BurstCompile]` test-suite attributes — already applied project-wide, so this spec is not
proposing that part again. What it *is* proposing is eliminating the trigger condition altogether
(no `Assert.*` inside `Execute` at all) rather than continuing to rely on staying inside an
unwritten "only bare bool" rule.

### 1.4 The good pattern, already established in this codebase

Two existing idioms both push assertion and formatting to the managed `[Test]` method, reading a
recorded verdict from the job:

**Best-in-class (zero `Assert.*` anywhere inside the struct)**:
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/ConvergenceBudgetTests.fProxy.cs`. The job
writes raw scalars into `NativeArray<int> Out` (`Store()`, lines 85-99); the managed `RunCase`
helper (lines 330-342) does everything else, including string formatting and both asserts:

```csharp
// job side (Execute(), no assert):
void Store(in SVDInfo info, int budget)
{
    Out[0] = info.Solved ? 1 : 0;
    Out[1] = info.sweeps;
    Out[2] = budget;
    Out[3] = info.converged;
}

// managed side, after .Run():
static void RunCase(TestJob.TestType type, int n, string label)
{
    var res = new NativeArray<int>(4, Allocator.TempJob);
    new TestJob { Type = type, N = n, Out = res }.Run();
    int solved = res[0], sweeps = res[1], budget = res[2], converged = res[3];
    res.Dispose();
    string line = $"[ConvBattery] {label} n={n} status={(solved != 0 ? "Converged" : "MaxIterations")} ...";
    Assert.IsTrue(solved != 0, line + " -- DID NOT CONVERGE");
    Assert.LessOrEqual(sweeps, budget / 4, line + " -- EXCEEDED 1/4 BUDGET MARGIN");
}
```

**Near-clean, in production battery use today** (`Fail[]`-array idiom, used across ~56 files):
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/PreconditionerBatteryTests.fProxy.cs`,
`KrylovSquareBatteryTests.fProxy.cs`, `KrylovBlockBatteryTests.fProxy.cs`,
`KrylovLstsqBatteryTests.fProxy.cs`, `KrylovBlockLstsqBatteryTests.fProxy.cs`. `Record()` writes
`(matrixIdx, checkId, got, expected)` into a shared `NativeArray<fProxy> Fail` on first failure, and
the managed `[TestCaseSource]` wrapper reads it and calls `Assert.Fail($"...")` with full context —
e.g. `PreconditionerBatteryTests.fProxy.cs:90-97,271-284`, `KrylovSquareBatteryTests.fProxy.cs:220-231,236-247`.
**Caveat**: every one of these `Record()` helpers *also* keeps a bare `Assert.IsTrue(ok)` inside the
job as an immediate-abort safety net (`PreconditionerBatteryTests.fProxy.cs:96`,
`KrylovSquareBatteryTests.fProxy.cs:230`, `KrylovBlockBatteryTests.fProxy.cs` `Record()` around
line 470) — so these files are in the "mixed" bucket of §1.1, not the fully clean one. Per §1.2 this
specific overload is believed Burst-safe, but §3 recommends dropping it anyway for consistency and
to remove the last managed call from the job.

**Deliberately no-`IJob` at all** (explicit design choice, not an oversight — acknowledged in each
file's header comment): `LOBPCGSmokeTests.fProxy.cs:17` ("Managed [Test]s (main thread), matching
the simpler non-Burst-job test style"), `LOBPCGRobustnessTests.fProxy.cs:17`,
`KrylovVerifyAtExitTests.fProxy.cs:22-23`. These run entirely as ordinary C# on the main thread —
correct and readable for "iteration-heavy, algorithm-level comparison" tests, but they forgo
Burst-path coverage entirely by design. Do not "fix" these into jobs as part of this spec's
conversion (§4) unless a specific Burst-only bug risk is identified for that file; that is a
separate judgement call the file's own author already made explicitly.

## 2. Detection method — get a real per-test Burst-vs-Mono verdict headlessly

Do not trust "no BC#### line in the log" as proof a job ran Burst-compiled — async/fallback
semantics have already surprised this project once (§1.3). Two complementary techniques:

### 2.1 `[BurstDiscard]` probe (per-job, deterministic, no Unity source changes needed)

`Unity.Burst.BurstDiscardAttribute` marks a method whose calls are **elided entirely from
Burst-compiled IL** but which executes normally under interpreted/Mono execution. This is the
standard Unity idiom for "is this code path actually running Burst-native right now," and gives an
unambiguous yes/no with no reliance on log scraping. Existing prior art for the attribute itself
(unrelated use) is at `Assets/LinearAlgebra/CodeGen/TemplateSource/proxyStructs.cs:103`.

Add a tiny, test-only shared helper (new file, e.g.
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/BurstProbe.cs`, not templated — it is dtype-free):

```csharp
using Unity.Burst;
using Unity.Collections;

namespace LinearAlgebra.Tests
{
    /// Test-only: detects whether the calling IJob.Execute() actually ran Burst-compiled or fell
    /// back to Mono. Call Mark(ref ran) once near the top of Execute(); ranMono[0] stays 0 under a
    /// real Burst compile (the call is elided) and becomes 1 under any Mono fallback.
    public static class BurstProbe
    {
        [BurstDiscard]
        public static void Mark(ref NativeArray<byte> ranMono) => ranMono[0] = 1;
    }
}
```

Any existing `[BurstCompile] IJob` can add one field + one call to self-report; this is meant as a
one-off diagnostic tool run against a *sample* of files during the conversion (§4), not a
permanently-shipped field on every test job.

### 2.2 Editor-log parsing for genuine compile-time errors (BC####)

`Tools/_unity-common.ps1:102-109` (`Get-CompileErrors`) already greps the Unity log for
`error CS\d+` (C# compile errors) after `Tools/run-tests.ps1`. Extend this pattern (do not modify
`run-tests.ps1`'s control flow, add a sibling check) with a second grep for Burst diagnostics —
`"Burst error BC\d+"` and `"burst_abort"` — over the same `TestResults/EditMode.log`. Any hit is
either (a) a genuine compile failure that should be loud (verify it correlates with a *failed* test
in the XML, confirming the `CompileSynchronously=true` memory claim from §1.2 point 4), or (b) a
runtime assert abort (expected, and *should* correlate with a failed test).

This gives the implementer, before doing any conversion work, a **verified** (not assumed) list of
which test files today rely on the runtime-assert-abort path (§1.2) vs. which (if any) are silently
falling back. Run this once as a baseline before §4 starts, and re-run after each stage to confirm
no regressions.

### 2.3 What NOT to rely on

`Assert.Fail($"…")` or any interpolated-string assert *inside* `Execute()` is a BC1071 compile
error per §1.2 — do not add one "just to test the detection method." Use the `BurstProbe` (§2.1)
for controlled experiments instead.

## 3. Prescribed pattern

`Execute()` performs pure Burst-compatible computation and records outcomes as raw scalars
(`bool`→`byte`/`int`, `fProxy`, enum→`int`) into a `NativeArray`/`NativeReference` field. The
`[Test]` method (or a shared managed helper, matching the existing `RunCase`/`Record`-reader idiom)
reads that array after `.Run()` and performs **every** `Assert.*` call and all string formatting
there.

Concretely, generalize the `ConvergenceBudgetTests` shape (§1.4) as the house style, and additionally
drop the belt-and-suspenders in-job `Assert.IsTrue(ok)` from the `Fail[]`-array idiom (§1.4's
caveat), so the pattern becomes fully clean rather than "believed safe":

```csharp
// BEFORE (bad -- Assert inside Execute, [Test] is a bare .Run()):
[BurstCompile(CompileSynchronously = true)]
public struct FooTestJob : IJob
{
    public void Execute()
    {
        var result = DoTheThing();
        Assert.IsTrue(result.ok);                     // <-- inside the job
        Assert.IsTrue(Analysis.isZero(result.residual, Tol()));
    }
}
[Test] public void FooTest() => new FooTestJob().Run();

// AFTER (record outside, matching ConvergenceBudgetTests/RunCase):
[BurstCompile(CompileSynchronously = true)]
public struct FooTestJob : IJob
{
    // [0] ok (1/0), [1] residual
    public NativeArray<fProxy> Out;
    public void Execute()
    {
        var result = DoTheThing();
        Out[0] = result.ok ? (fProxy)1 : (fProxy)0;
        Out[1] = result.residual;
    }
}
[Test]
public void FooTest()
{
    var res = new NativeArray<fProxy>(2, Allocator.TempJob);
    new FooTestJob { Out = res }.Run();
    bool ok = res[0] != (fProxy)0; fProxy residual = res[1];
    res.Dispose();
    Assert.IsTrue(ok);
    Assert.IsTrue(Analysis.isZero(residual, Tol()));   // full context available here, freely
}
```

For battery-style files that already use the `Fail[]` idiom, the only change is: delete the trailing
`Assert.IsTrue(ok)` line from `Record()`/`RecordBound()` (keep the `Fail[...]` writes) — the managed
`[TestCaseSource]` wrapper already reads `Fail[0]` and calls `Assert.Fail($"...")`, so nothing else
changes. This is a single-line deletion per `Record()`-shaped helper (§4 Stage 0).

**Codegen/`fProxy` interaction**: no template mechanics change. `NativeArray<fProxy>` /
`NativeArray<int>` fields expand per numeric type exactly like any other `fProxy`-templated field
already does (see `ConvergenceBudgetTests.fProxy.cs` — `NativeArray<int> Out` needs no `fProxy`
substitution since `int` is dtype-independent; `PreconditionerBatteryTests.fProxy.cs`'s
`NativeArray<fProxy> Fail` does substitute). No new `//+choose[...]` markers are needed for this
pattern. `CompileSynchronously = true` stays on every `[BurstCompile]` attribute (already there,
§1.3) — this spec does not touch that.

**Multiple/looped failures**: where a battery loops over many gallery matrices per `Execute()` call
(e.g. `RunStandardChecks`), keep the existing "record only the first failure" behavior (`if (!ok &&
Fail[0] == 0)`) — do not attempt to collect every failure in this pass; that is a separate,
unscoped improvement (§8 out of scope).

## 4. Staged conversion plan

Order by leverage (shared-helper files first, since fixing one `Record()` helper fixes every caller
in that file for free) and by risk (mechanical files first, files with `throw`/managed-exception
tests last, since those need the most care to keep `Assert.Catch`-style tests correct).

- **Stage 0 — battery `Record()`/`RecordBound()` helpers** (~6 files, one-line deletion each):
  `PreconditionerBatteryTests.fProxy.cs`, `KrylovSquareBatteryTests.fProxy.cs`,
  `KrylovBlockBatteryTests.fProxy.cs`, `KrylovLstsqBatteryTests.fProxy.cs`,
  `KrylovBlockLstsqBatteryTests.fProxy.cs`, `KrylovGridTests.fProxy.cs`, `SolverBatteryTests.fProxy.cs`,
  `LQRPTests.fProxy.cs` (`RecordBound`, line 1205). Delete the trailing `Assert.IsTrue(ok)` from each
  `Record`/`RecordBound`. Verify via `Tools/run-tests.ps1 -Filter "*Battery*"` (and `*LQRP*`) that the
  suite stays green — a real regression here means a check that was previously caught by the in-job
  abort is not actually being surfaced by the managed wrapper, which would itself be a pre-existing
  bug worth fixing at the same time.
- **Stage 1 — fully-bad files (43 files, §1.1)**: mechanical `.cs` files where `Type`-switch +
  bare-`.Run()` `[Test]`s dominate (`ConjugateGradientTests`, `CHOTests`, `CompareTests`,
  `SpecialConstructorsTests`, `LUTests`, `SVDTests`, `SparseSolverTests`, `TFQMRTests`,
  `DotOperationTests`, etc. -- full list obtainable by re-running the §1.1 scan). Convert one file at
  a time; each file becomes its own commit-sized unit. Use the §2.1 probe on the first 3-4 converted
  files to positively confirm they now run Burst-compiled (not just "still green"), then trust the
  pattern for the rest.
- **Stage 2 — mixed files (89 files)**: same conversion, lower urgency (already partially outside).
  Natural to fold into unrelated future edits of these files rather than a dedicated pass.
- **Not in scope for conversion**: the 7 already-clean files (§1.1), and the deliberately-no-`IJob`
  files (§1.4) unless a specific Burst-only bug is suspected there.

**Regression guard (do this as part of Stage 0, not as a follow-up)**: add
`Tools/check-burst-test-hygiene.ps1`, modeled directly on the existing
`Tools/check-doc-leaks.ps1` (regex scan, exit 1 on hits, `-ShowAll` switch). Scope: every `.cs` under
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/**` and `Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/**`.
Flag any `Assert.` occurrence whose enclosing braces are inside a `struct … : IJob`/`IJobParallelFor`
body (reuse the brace-matching approach used for the §1.1 survey scan) — this is a text-based
heuristic (false positives on nested local functions are acceptable; false negatives are the thing to
avoid). Print `path:line`. Wire it as an optional manual gate initially (documented in the script
header how to run it), not a blocking pre-commit hook, since this project has no CI yet
(`release-engineering-plan` memory: CI is still a v1.0 blocker).

## 5. Benchmark hygiene

Same risk, worse consequence: a benchmark job that silently falls back to Mono doesn't fail a test --
it reports a bogus timing number that looks plausible (just slow) and could stand unnoticed.

**Survey result**: `Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/*.cs` currently has **zero**
hits for `Debug.Log`, `Assert.*`, string interpolation (`$"`), `Enum.GetValues`, or `throw new` inside
any file. This family looks clean today.

**But this project has already been burned by exactly this class of bug once**, which is why the
survey above should not be read as "benchmarks are inherently safe" -- it's "the specific known
instance was fixed and nothing has regressed since." From
`Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/DEVLOG.md` (`## LPBenchmark`, 2026-07-11
entry): the report used to harvest `objective`/`iters`/`status` via *a separate plain managed call to
`LP.solve`/`LP.lad` before ever timing the Burst job* -- every benchmarked row solved the same problem
twice, once fully Mono-interpreted. Fine at `n=24`, but at `n=384` an extended run took minutes and
had to be killed. The fix moved reporting fields (`objOut`/`itersOut`/`statusOut`) into the timed job
itself, written from inside `Execute()` as a side effect of the already-timed native call. This is the
canonical concrete example of "a Mono-forced benchmark call measures the interpreter, not Burst, and
the numbers are meaningless" -- cite it as the model failure case in any benchmark-hygiene doc.

**Prescription for benchmarks** (lighter-touch than tests, since the survey found nothing currently
wrong):
1. Add the same `Assert.`/`Debug.Log`/`Enum.GetValues`/string-interpolation grep to
   `check-burst-test-hygiene.ps1` (§4) scoped additionally over `TemplateSourceBenchmarks/**`, so a
   future benchmark edit can't reintroduce this class silently.
2. For any *new* benchmark job going forward: report fields (objective/iters/status/whatever the
   table prints) must be written from inside the timed `Execute()`, never fetched via a second
   separate (potentially Mono) call outside timing -- codify this as the one-line rule in
   `Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/DEVLOG.md`'s top-level header or a short
   comment in `Assets/LinearAlgebra/Benchmarks/Bench.cs`, pointing at the LPBenchmark history entry as
   the "why."
3. Apply the §2.1 `BurstDiscard` probe to one or two benchmark jobs as a spot check (e.g.
   `GemmBenchmark.fProxy.cs`, `KrylovGridBenchmark.fProxy.cs`) to positively confirm current
   benchmarks run Burst-native, rather than relying solely on the "plausible-looking" timing numbers
   as evidence.

## 6. Test naming + battery integration

### 6.1 The concrete problem: `KrylovAuditRegressionTests.fProxy.cs`

`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovAuditRegressionTests.fProxy.cs` is a
new, **uncommitted** file (confirmed via `git status --porcelain`: `??` for the template and both
generated copies) with 5 test cases, one per audit-confirmed bug (minresQLP warm start, btfqmr
zero-row x2, BiCGStab verify-fail-continue, bgcrodr uninitialized `AUblk`). It is:

- Named after the *event that found it* ("audit regression"), not the *property it verifies* -- the
  project's own naming precedent for exactly this situation is
  `KrylovVerifyAtExitTests.fProxy.cs` (names the behavior under test -- "verify at exit" -- not "bug
  found in review round N"), consistent with the CLAUDE.md comment policy's ban on
  process/ticket/round references leaking into shipped artifacts (this is a test *file name*, an
  even more visible/permanent surface than a comment).
- Structurally isolated: it builds its own ad hoc SPD/nonsym matrix generators
  (`BuildDenseSPD`, `DenseNonsym`, `MatVec`, `RelResVec`, `RelResBlockRow` -- lines 63-121) that
  duplicate what `KrylovBattery.Gallery.fProxy.cs` + `fProxyKrylovBatteryOracles` already provide,
  and calls `Krylov.minresQLP`/`Krylov.btfqmr`/`Krylov.biCGStab`/`Krylov.bgcrodr` directly rather
  than through the `IfProxySquareSolverInvoker`/`IfProxyBlockSolverInvoker` struct-functor
  abstraction the rest of the battery uses (`KrylovBattery.Invokers.fProxy.cs`).
- Because it is untracked and it is the specific artifact the owner called unacceptable, it should
  be **dissolved into the battery**, not renamed in place -- see 6.3.

### 6.2 Existing naming convention (from files that ARE already committed/accepted)

- `Krylov<Shape>BatteryTests.fProxy.cs` (`KrylovSquareBatteryTests`, `KrylovBlockBatteryTests`,
  `KrylovLstsqBatteryTests`, `KrylovBlockLstsqBatteryTests`) -- cross-cutting solver x matrix
  coverage, driven by shape (square/block/lstsq/block-lstsq), one `[TestCaseSource]` enumerating
  solvers.
- `Krylov<Property>Tests.fProxy.cs` (`KrylovVerifyAtExitTests`, `KrylovFusedKernelTests`) -- a single
  cross-cutting *behavior* or *implementation property*, verified across the solvers it applies to.
- `<Solver>Tests.fProxy.cs` (`ConjugateGradientTests`, `GMRESTests`, `TFQMRTests`, `IDRTests`, ...) --
  one solver's own unit tests (API surface, edge cases specific to that solver, not battery-shaped).
- `PreconditionerBatteryTests.fProxy.cs` -- preconditioner x matrix cross-coverage, the sparse
  analogue of the solver batteries.

(`KrylovRound2Tests.fProxy.cs` is itself a pre-existing naming outlier -- a round-number reference,
the exact thing CLAUDE.md's comment policy forbids in comments -- but it is already committed and
out of scope for this spec; flag it as a candidate for a future rename-only cleanup, do not rename it
as part of this task.)

**Rule for new Krylov test files going forward**: name by *shape* (if it's a new battery dimension)
or by *property/behavior* (if it's a cross-cutting check like verify-at-exit) -- never by "regression
for bug X" or a review/audit event. A regression coverage need for a *specific, already-fixed* bug
that genuinely doesn't generalize into a standardized check (rare -- most audit findings, per 6.3,
generalize fine) should become one or two more `[TestCase]`s inside the relevant existing
`<Solver>Tests.fProxy.cs` file, with a comment citing the mechanism (not the audit), not a new file.

### 6.3 Folding the audit's scenario gaps into the battery as standardized checks

The audit's own language makes the generalization case directly: *"No test warm-starts minresQLP
(all use zeroed arena.fProxyVec)"* (audit doc, minresQLP finding) and *"Zero row padding makes the
trigger exact... other, perfectly solvable row fails too"* (audit doc, btfqmr finding) are both
**family-wide gaps**, not solver-specific ones -- nothing about warm-starting or zero-RHS rows is
particular to minresQLP or btfqmr; every square/block solver takes a caller-supplied `ref x`/`ref X`
(already the warm-start entry point -- see `IfProxySquareSolverInvoker.Solve<TOp>(in TOp A, in
fProxyN b, ref fProxyN x)`, `KrylovBattery.Invokers.fProxy.cs:23-24`) and every block solver takes a
caller-supplied `B` matrix whose rows are independent.

Add two new standardized checks, gated by `Requires`/`Forbids` exactly like the existing five
(`MatrixProfileMatch.Applicable`, `KrylovBattery.Invokers.fProxy.cs` / `KrylovBatteryProfile.cs`),
so a fix to the check catches the bug class everywhere at once instead of one solver:

- **Check #6 -- Warm-start correctness**, square battery
  (`KrylovSquareBatteryTests.fProxy.cs`, slotting into `RunStandardChecks`/`CheckDense`/`CheckBSR`
  right after check #5, i.e. checks 1-5 exist today per `CheckDense`/`CheckBSR`,
  lines 86-116/136-188). For each applicable gallery matrix: solve once from `x=0` to get a
  reference `xRef` (or reuse `xRef` from check #2's reference solve); seed `x` to a fixed nonzero
  vector unrelated to `xRef` (same recipe as the now-dissolved
  `KrylovAuditRegressionTests.MinresQLPWarmStartRecovered`); solve again with that warm start;
  `Record` that the warm-started result matches `xRef` within `tolBand` AND that the fresh residual
  (`fProxyKrylovBatteryOracles.RelResidualDense`/`RelResidualBSR`) is within the existing check #1
  band. Applies to every invoker unconditionally (no new `Requires`/`Forbids` needed -- every square
  solver takes `ref x`).
- **Check #10 -- Degenerate/zero RHS row does not break the batch**, block battery
  (`KrylovBlockBatteryTests.fProxy.cs`, in `CheckBlockAdditions`, after existing checks #6-#9,
  lines 402-440). Build a block RHS `B` with one row zeroed (mirrors
  `BtfqmrZeroRhsRowMixedBatch`); solve; `Record` that (a) status is never `Breakdown` (existing
  `flags.NoBreakdown` gate already expresses "this solver documents Breakdown as impossible here" --
  reuse it), (b) every *other* row still meets the check #1-equivalent residual band, (c) no
  NaN/Inf anywhere in `X` (mirrors existing check #9's NaN/Inf scan, line 426-430). This directly
  generalizes the btfqmr finding to bgmres/bminres/bbiCGStab/bidr/bgcrodr/etc. -- whichever of those
  the invoker's `Requires`/`Forbids` already lets the check reach.
- **Recycling-cycles-finite** (bgcrodr's bug #4) is narrower -- it's specific to solvers that keep
  cross-restart-cycle state (currently only `bgcrodr`/`gcrodr`). Do not force this into the two
  standardized checks above; instead add it as a small, separate, solver-specific
  `[TestCase]`-style addition inside `GCRODRTests.fProxy.cs`/`BlockGCRODRTests.fProxy.cs` (which
  already exist) using a small per-cycle Krylov budget to force multiple restart cycles (same
  construction `KrylovAuditRegressionTests.BgcrodrRecycleCyclesFinite` used, lines 289-325) -- but
  written in the record-outside pattern (§3), and with a comment citing "forces multiple recycling
  rebuilds" (the mechanism), never "audit finding" or "task #64" (the provenance).
- The BiCGStab `v`-corruption bug (verify-fail-continue path) is a **path guard**, not a
  scenario-coverage gap -- no gallery matrix/RHS combination deterministically triggers it, so it
  cannot become a `Requires`/`Forbids`-gated standardized check the same way. Fold it into
  `BiCGStabTests.fProxy.cs` (its existing per-solver file -- confirm exact filename before writing;
  it's the file backing `fProxyBiCGStabInvoker`) as an additional ill-conditioned-nonsymmetric case,
  asserting the same finite-and-honestly-converged outcome
  `KrylovAuditRegressionTests.BiCGStabIllConditionedNoStall` used (lines 255-280), again in the
  record-outside pattern.

**Disposition of the file itself**: once the four items above are folded in, delete
`KrylovAuditRegressionTests.fProxy.cs` (it is untracked -- no history to preserve) and its two
generated copies (deleted automatically by the next `Tools/regen.ps1` once the template is gone --
do not hand-delete `Assets/LinearAlgebra/SourceTests/Generated/**`, per CLAUDE.md).

## 7. Acceptance criteria

Diagnosis/detection tooling (this spec's own deliverables, testable independent of any conversion):

- [ ] `Tools/check-burst-test-hygiene.ps1` exists, follows `Tools/check-doc-leaks.ps1`'s
      conventions (exit 0 clean / exit 1 with `path:line` hits, `-ShowAll` switch, doc-comment header),
      and scans both `TemplateSourceTests/**` and `TemplateSourceBenchmarks/**`.
- [ ] Running it against the current tree (pre-conversion) reports a nonzero hit count consistent
      with §1.1's counts (roughly 132+ files); running it against a converted file (post-Stage-0/1)
      reports zero hits for that file.
- [ ] `BurstProbe.Mark` (§2.1) exists as a test-only, non-templated helper; a throwaway test using it
      against (a) a known-good `[BurstCompile] IJob` and (b) a job with `[BurstCompile(Enabled =
      false)]` (or an equivalent forced-Mono job) demonstrates `ranMono` stays 0 in case (a) and
      becomes 1 in case (b) -- i.e., the probe is proven to discriminate correctly, not merely
      asserted to.

Conversion (§3/§4), checkable per stage:

- [ ] Stage 0: every `Record`/`RecordBound` helper listed in §4 no longer contains a bare
      `Assert.IsTrue(ok)` call; `Tools/run-tests.ps1 -Filter "*Battery*"` and `-Filter "*LQRP*"` stay
      green.
- [ ] Stage 1 (per converted file): the file's `[Test]` methods contain the assertions (previously
      inside `Execute()`); `Execute()` only writes to a `NativeArray`/`NativeReference` output field;
      `check-burst-test-hygiene.ps1` reports zero hits for that file; the file's tests still pass
      under `Tools/run-tests.ps1 -Filter "<TypeName>"`.
- [ ] For at least 3 Stage-1-converted files, the §2.1 probe confirms Burst execution (not just "test
      still green").

Battery integration (§6):

- [ ] `KrylovSquareBatteryTests.fProxy.cs` gains check #6 (warm-start correctness), gated identically
      to checks #1-5, running for every applicable gallery matrix (dense + BSR).
- [ ] `KrylovBlockBatteryTests.fProxy.cs` gains check #10 (degenerate zero-RHS row), running for
      every applicable block solver via the existing `Requires`/`Forbids` gate.
- [ ] `KrylovAuditRegressionTests.fProxy.cs` (template) and its two generated copies no longer exist.
- [ ] `GCRODRTests.fProxy.cs`/`BlockGCRODRTests.fProxy.cs` gain the recycling-cycles-finite case;
      `BiCGStabTests.fProxy.cs` (verify exact filename) gains the ill-conditioned no-stall case.
- [ ] `Tools/run-tests.ps1 -Filter "*Krylov*"` (and `*GCRODR*`, `*BiCGStab*`) is green after the fold,
      and -- this is the actual point of §6.3 -- the new checks reach every applicable solver, not
      just the one the audit happened to flag (spot-check: temporarily seed a warm-start bug in a
      second solver, e.g. `cg`, confirm check #6 catches it too, then revert the seeded bug).
- [ ] No new test file name in this area contains an audit/round/ticket reference (grep for
      `Audit`, `Regression`, `Round\d`, `Task#?\d` in new file names as a final check).

## 8. Out of scope

- Do **not** run Unity or the test suite as part of producing this spec (already followed -- this was
  a read-only survey; a separate test-writer/coder agent runs headless afterward).
- Do not attempt to make batteries collect *every* failure in a run (removing the `Fail[0] == 0`
  first-failure-only guard) -- that's a genuine, separate improvement or a Burst limitation to work
  around, and is not required to fix the hygiene problem this spec targets.
- Do not convert the deliberately-no-`IJob` files (`LOBPCGSmokeTests`, `LOBPCGRobustnessTests`,
  `KrylovVerifyAtExitTests`) into jobs -- that was an explicit, documented author choice, not an
  instance of this hygiene problem.
- Do not rename `KrylovRound2Tests.fProxy.cs` (pre-existing naming debt, noted in §6.2 but not
  actioned here) or touch any other already-committed file's name beyond what §6.3 requires.
- Do not add `[Parallelizable]` or any other suite-speed mechanism the project has already rejected
  (see `faster-testing-task` project memory's "OUT OF SCOPE" list) -- this spec is about correctness
  coverage and hygiene, not another speed pass.
- Do not build CI/pre-commit enforcement for `check-burst-test-hygiene.ps1` -- ship it as a manually
  run tool only; this project has no CI yet (a separate, tracked v1.0 blocker).
- Open question for the owner, not resolved here: whether the §1.2-point-4 claim (BC compile errors
  under `CompileSynchronously=true` reliably throw rather than silently fall back) should be
  formally re-verified end-to-end (deliberately introducing a BC1330/BC1071 in a scratch test file,
  running the suite, confirming a loud failure, then reverting) before Stage 1 begins, or whether the
  existing 2026-07-06 memory + this spec's `burst_abort` runtime-assert evidence (§1.2) is trusted as
  sufficient. Recommend doing the one-off verification (cheap, ~10 minutes) since Stage 1's whole
  value proposition rests on it.
