# Release scan 2026-07-12 — area: tools-scripts (non-template)

Scanned 8 files (infra). Counts: total 5, confirmed 5, uncertain 0, unverified 0, refuted 0 — high 0, medium 2, low 3.

## Scope

- Tools/_unity-common.ps1
- Tools/benchmark.ps1
- Tools/check-doc-leaks.ps1
- Tools/clean-generated.ps1
- Tools/prune-orphaned-generated.ps1
- Tools/regen-and-test.ps1
- Tools/regen.ps1
- Tools/run-tests.ps1

## Findings

### 1. [medium/logical/CONFIRMED] Tools/regen.ps1:89 — -Check drift check includes the CodeGen template (INPUT) tree in git-status, so uncommitted template edits produce a spurious drift failure with a message that misattributes them to generated files.

**Evidence**

```
Line 89: git status --porcelain -- "Assets/LinearAlgebra/Source" "Assets/LinearAlgebra/SourceTests" "Assets/LinearAlgebra/CodeGen" "Assets/LinearAlgebra/Benchmarks/Generated".
```

regen/CodegenBootstrap NEVER writes to CodeGen (Program.cs only writes Source, SourceTests/Generated, Benchmarks/Generated), so any diff reported under CodeGen is a pre-existing uncommitted template edit, not codegen output. In the documented pre-commit use, that makes -Check fail (line 100-101) with 'generated files were out of sync with templates. Commit the regenerated files.' even when generated output is perfectly in sync.

**Verifier**

Confirmed by direct inspection.

Line 89 of Tools/regen.ps1 (verbatim):
`$changed = @(git status --porcelain -- "Assets/LinearAlgebra/Source" "Assets/LinearAlgebra/SourceTests" "Assets/LinearAlgebra/CodeGen" "Assets/LinearAlgebra/Benchmarks/Generated" 2>$null)`

The result feeds `$changed.Count -gt 0` (line 92) and, when `-Check` is set, prints "generated files were out of sync with templates. Commit the regenerated files." and exits 1 (lines 99-101).

Verification that CodeGen is INPUT-only for the generator:
- Tools/CodegenBootstrap/Program.cs (lines 36-50) writes only to three targets, taken from GenUtils constants:
  - `GenUtils.generatedFolder` = "Assets/LinearAlgebra/Source/"
  - `GenUtils.generatedTestsFolder` = "Assets/LinearAlgebra/SourceTests/Generated/"
  - `GenUtils.generatedBenchmarksFolder` = "Assets/LinearAlgebra/Benchmarks/Generated/"
- Program.cs `WriteScriptsFromContext` (lines 103-147) uses `context.overrideFolderPath`, which is set to one of those three outputs; nothing under Assets/LinearAlgebra/CodeGen is ever written by the bootstrap.
- Tools/prune-orphaned-generated.ps1 lines 108-110 confirm the same three (Template, Generated) pairs — its Generated targets never include CodeGen.

Contents of Assets/LinearAlgebra/CodeGen/ are all hand-authored or non-code artifacts: GenUtils.cs, TemplateConverter.cs, TemplateSource*Generator.cs, the three Template* input trees (with DEVLOG.md files inside per CLAUDE.md's policy), plus asmdef files.

Concrete failing scenarios:
1. Author edits `Assets/LinearAlgebra/CodeGen/TemplateSource/.../DEVLOG.md` (never read by codegen — CLAUDE.md: "codegen only reads *.cs"). `regen.ps1 -Check` regenerates zero output changes, yet git status still shows DEVLOG.md as modified under the CodeGen pathspec. `$changed.Count > 0` and the script exits 1 with "generated files were out of sync with templates" — pure false positive; regenerating cannot fix it.
2. Author edits `GenUtils.cs` or `TemplateConverter.cs` in a way that does not change output (comment-only, refactor). Same false positive.
3. Author edits a template AND has already run regen so generated output matches. Both template and generated files show unstaged in git status; the drift message reads "generated files were out of sync… Commit the regenerated files" — misattributes the diff (the generated tree is in sync).

The `Assets/LinearAlgebra/SourceTests` pathspec is also arguably too broad (root also contains hand-written tests like ArenaConcurrencyTests.cs, QPSolveTests.cs etc.), which would similarly trigger false-positive drift on unrelated test edits — this is an adjacent bug the fix should address, though the reported claim is specifically about the CodeGen entry.

Recommended fix (as claim states): drop `Assets/LinearAlgebra/CodeGen` from the pathspec; the drift check should only cover the three actual output roots. Also narrow `Assets/LinearAlgebra/SourceTests` to `Assets/LinearAlgebra/SourceTests/Generated` to close the analogous false-positive on hand-written tests. Alternatively, disambiguate the failure message so template-tree diffs are reported as "uncommitted template edits" vs "generated drift".

File path (absolute): C:\Users\viliv\Documents\LinearAlgebraBursted\Tools\regen.ps1 (line 89), corroborated by C:\Users\viliv\Documents\LinearAlgebraBursted\Tools\CodegenBootstrap\Program.cs (lines 36-50), C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\CodeGen\GenUtils.cs (lines 53-59), C:\Users\viliv\Documents\LinearAlgebraBursted\Tools\prune-orphaned-generated.ps1 (lines 108-110).

**Suggested fix**

Drop "Assets/LinearAlgebra/CodeGen" from the git status path list used for the drift/-Check decision (keep only the three generated trees), or separate 'template changed' from 'generated drift' in the message.

### 2. [medium/pointer/CONFIRMED] Tools/benchmark.ps1:91 — Benchmark results table (the script's deliverable) is read with Get-Content, which in PS 5.1 mis-decodes BOM-less UTF-8 (this repo was bitten by exactly this), corrupting any non-ASCII in the printed table.

**Evidence**

```
Line 91: `Get-Content $Results` echoes the results file that the benchmark wrote via .NET File.WriteAllText (UTF-8, no BOM per CodegenBootstrap's own note).
```

PS 5.1 Get-Content defaults to the system ANSI code page for BOM-less files, so µs / × / ² style glyphs mojibake on output. Contrast check-doc-leaks.ps1:63 and prune-orphaned-generated.ps1:75 which correctly use [System.IO.File]::ReadLines/ReadAllText. Same pattern in run-tests.ps1:83/99 (failure messages).

**Verifier**

Verified defect. Bench.cs:113 writes TestResults/benchmark-all.txt via File.WriteAllText (UTF-8 without BOM by .NET default). The report body contains many non-ASCII glyphs written by sb.AppendLine in FFTBenchmark, KernelBenchmark, IterativeBenchmark, KMeansBenchmark, etc. (arrows →, em-dash —, middle-dot ·, transpose superscript ᵀ on Aᵀ/Mᵀ/Cᵀ). benchmark.ps1:91 reads this file with bare `Get-Content $Results`, which in PS 5.1 defaults to the system ANSI code page for BOM-less files, mojibaking those glyphs in the printed table. Sibling scripts (check-doc-leaks.ps1:63 uses [System.IO.File]::ReadLines; prune-orphaned-generated.ps1:75 uses [System.IO.File]::ReadAllText) already do this correctly, and the repo's MEMORY explicitly flags the PS 5.1 UTF-8 trap. The specific glyphs cited in the evidence (µ, ×, ²) are illustrative rather than currently emitted; actual mojibake affects →, —, ·, Aᵀ/Mᵀ/Cᵀ — same fix applies. The file on disk remains valid UTF-8; only console echo is corrupted. Suggested fix (swap to [System.IO.File]::ReadAllText or Get-Content -Encoding UTF8) is correct. Note the claim's extension to run-tests.ps1 is partially right: :83 [xml] cast honors the encoding declaration and is fine, but the Get-Content $Log -Tail reads at :78/:103 do exhibit the same trap on Unity editor logs.

**Suggested fix**

Read with [System.IO.File]::ReadAllText($Results) (or Get-Content -Encoding UTF8) before echoing; likewise for the XML/failure-message reads in run-tests.ps1.

### 3. [low/naming/CONFIRMED] Tools/clean-generated.ps1:3 — Synopsis claims it deletes ALL codegen output but the target list omits the Benchmarks/Generated tree, which is genuine codegen output (46 committed files).

**Evidence**

```
Lines 2-3: 'Deletes ALL codegen output (Source, SourceTests/Generated)'; description line 18 'delete everything under the two Generated/ trees'; $targets (lines 37-40) list only $sourceRoot and SourceTests/Generated.
```

But Assets/LinearAlgebra/Benchmarks/Generated exists, is produced by CodegenBootstrap Program.cs pair 3 and pruned by prune-orphaned-generated.ps1 pair 3. A 'full clean' therefore leaves the benchmark tree untouched (harmless in practice only because regen.ps1 later runs prune).

**Verifier**

clean-generated.ps1 synopsis (line 3) and description (line 17-18) claim to delete "ALL codegen output" / "everything under the two Generated/ trees", but $targets (lines 37-40) only enumerates Source + SourceTests/Generated. The Benchmarks/Generated tree is a genuine, populated codegen output (46 committed .cs files produced from TemplateSourceBenchmarks): prune-orphaned-generated.ps1:107-111 lists it as its third template->generated pair, and regen.ps1:89 tracks it as an output. So the synopsis' "ALL" is factually inaccurate. Impact is limited because regen.ps1 invokes prune-orphaned-generated.ps1 first (which does cover all three trees), so a stock clean->regen still ends up consistent — hence low severity naming/documentation defect, not a functional bug. Suggested fix: either add a third $targets entry for Benchmarks/Generated, or reword the synopsis to state the two-tree scope explicitly.

**Suggested fix**

Add the Benchmarks/Generated tree to $targets (Keep = @()), or reword the synopsis to state it intentionally covers only Source + SourceTests.

### 4. [low/naming/CONFIRMED] Tools/regen.ps1:13 — Description lists only two template->output pairs but the headless bootstrap actually generates three (benchmarks included).

**Evidence**

```
Lines 11-16 describe 'TemplateSource -> Source, TemplateSourceTests -> SourceTests/Generated' and 'the project's two [Generator] wrappers'.
```

CodegenBootstrap/Program.cs (lines 46-49) processes a third pair: GenUtils.sourceBenchmarksTemplateFolder -> generatedBenchmarksFolder, and regen's own drift check (line 89) and prune (pair 3) both cover Benchmarks/Generated. The doc undercounts.

**Verifier**

Verified against Assets/LinearAlgebra/CodeGen/: there are THREE [Generator] wrappers (TemplateSourceGenerator.cs, TemplateSourceTestsGenerator.cs, TemplateSourceBenchmarksGenerator.cs), each pointing at a distinct source/output pair in GenUtils.cs (sourceTemplateFolder->generatedFolder, sourceTestsTemplateFolder->generatedTestsFolder, sourceBenchmarksTemplateFolder->generatedBenchmarksFolder). Tools/CodegenBootstrap/Program.cs lines 36-50 correctly enumerates all three pairs. But Tools/regen.ps1 lines 9-11 DESCRIPTION says the bootstrap "runs them the same way the project's two [Generator] wrappers do (TemplateSource -> Source, TemplateSourceTests -> SourceTests/Generated)" - undercounts by one and omits TemplateSourceBenchmarks -> Benchmarks/Generated. Internal inconsistency confirmed: regen.ps1's own drift check on line 89 does include Assets/LinearAlgebra/Benchmarks/Generated, so the script knows about the third pair; only the DESCRIPTION doc block is stale. Line number in report (13) is off - the misleading text is actually on lines 9-11 - but the defect itself is real. Severity is correctly rated low: doc-only drift, no execution impact. As a bonus, the same undercount exists in Program.cs lines 10-13 comment. Suggested fix (mention the benchmarks pair) is appropriate.

**Suggested fix**

Mention the TemplateSourceBenchmarks -> Benchmarks/Generated pair in the .DESCRIPTION.

### 5. [low/logical/CONFIRMED] Tools/run-tests.ps1:88 — A -Filter that matches zero tests yields a NUnit run with result=Passed/total=0, so the script reports success even though nothing ran.

**Evidence**

```
Line 63 converts the glob to a regex; if it matches no test the run still completes with result='Passed'. Line 88 `if ($run.result -eq "Passed") { exit 0 }` treats that as full success, and line 85 prints total=0 without warning.
```

A typo'd filter silently 'passes'.

**Verifier**

Verified against Tools/run-tests.ps1 (whole file) and Tools/_unity-common.ps1.

Code path confirmed:
- Line 60-65: `$Filter` glob is regex-escaped only via `-replace '\*', '.*'`; a typo like "*Egien*" becomes valid regex `.*Egien.*`, so Unity accepts it and simply matches no tests.
- Line 67: Unity's exit code is captured into `$exit` but never used as a gate — the docstring explicitly says the script "trusts the NUnit XML result file rather than Unity's exit code, which is unreliable across versions."
- Line 72-81: The only pre-XML guard is `if (-not (Test-Path $Results))`; it fires only when the file is absent.
- Line 83-86: XML parsed; `total`, `passed`, `failed`, `skipped` printed on one info line with no warning-marker even when they are all zero.
- Line 88: `if ($run.result -eq "Passed") { exit 0 }` — the only success predicate. No secondary check on `$run.total`, `$run.testcasecount`, or `$run.passed`.
- _unity-common.ps1 has no post-run XML validation either.

XML format cross-checked against the actual TestResults/EditMode.xml on disk: `<test-run ... result="Passed" total="5879" passed="5879" ...>` — `result` and `total` are independent attributes in the NUnit v3 schema. Unity's Test Framework (NUnit v3 wire format) produces `result="Passed"` when zero tests match a `-testFilter`; this is documented NUnit-v3 behavior and matches what CI logs show.

Concrete failing scenario: `./Tools/run-tests.ps1 -Filter "*Egien*"` (typo). Regex matches zero tests. Unity produces EditMode.xml with `result="Passed" total="0"`. Script prints `Result=Passed total=0 ...` and exits 0. A CI job or `/loop` cycle would treat this as green while nothing was verified.

No documented contract disclaims this; the parameter help text presents `-Filter` as a plain matcher with no mention of the empty-match hazard.

Suggested fix (fail or at least warn when `[int]$run.total -eq 0`) is a one-liner before line 88 and lossless.

Severity "low" is fair — bite only on user typo, and total is printed — but the defect is real: a gating script reports success when zero tests ran.

File: C:\Users\viliv\Documents\LinearAlgebraBursted\Tools\run-tests.ps1 (line 88 the exit-0 path; line 63 the glob rewrite; line 85-86 the silent total print).

**Suggested fix**

After parsing, also fail (or at least warn) when [int]$run.total -eq 0.

## Scanner notes

Verified against non-assigned files for correctness only (findings reported against assigned .ps1 files): Assets/LinearAlgebra/CodeGen/TemplateConverter.cs and GenUtils.cs (prune mapping mirror is faithful), Tools/CodegenBootstrap/Program.cs (3 template->output pairs, UTF-8 no BOM writes). Explicitly checked and found CORRECT: benchmark.ps1 affinity mask math (upper/lower half bit ranges right; [long]1 -shl $i avoids int32 sign-extension into the 64-bit mask); prune-orphaned-generated.ps1 singular/fProxy/iProxy/alsoExpand mapping matches the converter's filename-based decision; regen callers (benchmark.ps1:44, run-tests.ps1:41) correctly check $LASTEXITCODE after `& regen.ps1`, and regen checks $LASTEXITCODE after both `dotnet build` and the bootstrap run; run-tests.ps1 deletes stale $Results before running (benchmark.ps1 relies on a fresh -logFile so no stale-file false pass). No high-severity defects found.
