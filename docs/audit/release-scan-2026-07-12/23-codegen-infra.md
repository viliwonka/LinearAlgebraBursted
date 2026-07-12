# Release scan 2026-07-12 — area: codegen-infra (non-template)

Scanned 6 files (infra), counts: {"total":3,"confirmed":3,"uncertain":0,"unverified":0,"refuted":0,"high":0,"medium":1,"low":2}

## Scope

- Assets/LinearAlgebra/CodeGen/GenUtils.cs
- Assets/LinearAlgebra/CodeGen/TemplateConverter.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceGenerator.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceTestsGenerator.cs
- Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarksGenerator.cs
- Tools/CodegenBootstrap/Program.cs

## Findings

### 1. [medium/logical/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateConverter.cs:554 — copyReplaceAll caps-token array uses lowercase boolTypes instead of capsBoolTypes, so the bool row of an allTypes block substitutes the FProxy caps token with "bool" instead of "Bool".

**Evidence**

```
Line 553 builds types with `...Concat(GenUtils.boolTypes)` (correct), but line 554 builds capsTypes with `GenUtils.capsFloatTypes.Concat(GenUtils.capsIntTypes).Concat(GenUtils.boolTypes)` - it concatenates boolTypes ({"bool"}) where every other slot uses the caps array. GenUtils.capsBoolTypes = {"Bool"} (GenUtils.cs:51) is defined but never referenced (grep confirms only its declaration). For the bool iteration in GenerateForAllTypes, capsTypes[i] is therefore "bool", so any FProxy caps token inside a //+copyReplaceAll block emits e.g. boolPivot instead of BoolPivot.
```

**Verifier**

Verified against source. TemplateConverter.cs:554 builds `capsTypes` for the allTypes branch as `capsFloatTypes.Concat(capsIntTypes).Concat(GenUtils.boolTypes)` — the last term is the lowercase array `{"bool"}` where the intended caps array `GenUtils.capsBoolTypes = {"Bool"}` (declared at GenUtils.cs:51, otherwise unreferenced anywhere in the tree) should sit. Line 580 uses `capsTypes[i]` as the substitution for the `FProxy` caps token (`cFProxy = "FProxy"` per GenUtils.cs:41), so for the bool iteration in a `//+copyReplaceAll` block any `FProxy`-caps token would emit `bool<...>` instead of `Bool<...>` (e.g. `boolPivot` rather than `BoolPivot`). Latent today: `CopyReplaceAll` (line 243, the only caller with `allTypes=true`) processes only Pivot.Operations.cs, and that template uses only lowercase `fProxy`/`fProxyN`/`fProxyMxN` tokens — no FProxy caps token to miscap. No downstream guard rewrites `"bool"`→`"Bool"`. No contract, no intentional design justifies the discrepancy. Suggested fix (`.Concat(GenUtils.capsBoolTypes)`) is correct. Claim is a real, latent codegen defect.

**Suggested fix**

Change line 554 to `.Concat(GenUtils.capsBoolTypes)`. Latent today (Pivot.Operations.cs, the only copyReplaceAll file, uses only lowercase fProxy tokens, not the FProxy caps token), but a future copyReplaceAll template using the caps token would silently miscap the bool variant.

### 2. [low/naming/CONFIRMED] Tools/CodegenBootstrap/Program.cs:11 — Header comment claims the bootstrap runs 'two [Generator] wrappers' but it processes three source/output pairs (source, tests, and benchmarks).

**Evidence**

```
Lines 11-13 say 'runs the project's two [Generator] wrappers (TemplateSourceGenerator, TemplateSourceTestsGenerator)'. The pairs array at lines 36-50 has three entries, the third being sourceBenchmarksTemplateFolder -> generatedBenchmarksFolder, matching the third real [Generator], TemplateSourceBenchmarksGenerator. The doc is stale relative to the code.
```

**Verifier**

Absolute path: C:\Users\viliv\Documents\LinearAlgebraBursted\Tools\CodegenBootstrap\Program.cs

Lines 11-13 state the bootstrap "runs the project's two [Generator] wrappers (TemplateSourceGenerator, TemplateSourceTestsGenerator)". But the pairs array on lines 36-50 has three entries, the third being (GenUtils.sourceBenchmarksTemplateFolder, GenUtils.generatedBenchmarksFolder) on lines 46-49, and the foreach loop on line 53 processes all three unconditionally. Confirmed cross-reference: Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarksGenerator.cs exists as a real [Generator] wrapper alongside TemplateSourceGenerator and TemplateSourceTestsGenerator, and it maps to exactly that third pair (targetBasePath = GenUtils.generatedBenchmarksFolder; converter.Execute(context, GenUtils.sourceBenchmarksTemplateFolder)). There is no guard, no scoping caveat, and no "these two are the primary ones" framing in the comment — it plainly miscounts. Severity is appropriately low (doc-only, no runtime impact), but the defect is real: the comment was not updated when the benchmarks wrapper was added. Suggested fix: rewrite the header to say "three [Generator] wrappers" and add TemplateSourceBenchmarksGenerator to the parenthetical list.

**Suggested fix**

Update the comment to say three wrappers and add TemplateSourceBenchmarksGenerator to the list.

### 3. [low/naming/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateConverter.cs:192 — Infinity-guard error messages in CopyReplaceFill and CopyReplaceAll both say 'copyReplace syntax is bad', mislabeling which marker family failed.

**Evidence**

```
CopyReplaceFill line 192: Debug.LogError("Infinity guard triggered, copyReplace syntax is bad: ...") - but this method handles //+copyReplaceFill. CopyReplaceAll line 251 has the identical 'copyReplace syntax is bad' text for //+copyReplaceAll. Only DeleteThis/ChooseReplace/SkipForReplace name their own marker.
```

**Verifier**

Verified by direct reading of C:\Users\viliv\Documents\LinearAlgebraBursted\Assets\LinearAlgebra\CodeGen\TemplateConverter.cs. CopyReplaceFill (method starts line 163, guards `//+copyReplaceFill` via GenUtils.copyFillMarkerStart) logs "copyReplace syntax is bad" at line 192; CopyReplaceAll (method starts line 230, guards `//+copyReplaceAll` via GenUtils.copyAllMarkerStart) logs the same "copyReplace syntax is bad" at line 251. Only the middle CopyReplace method (line 197, guards `//+copyReplace` via GenUtils.copyMarkerStart) is actually accurate at line 218. Sibling methods DeleteThis (line 269: "deleteThis syntax is bad") and ChooseReplace (lines 290/298: "choose marker missing ...") do name their own markers, so per-marker labeling is the established pattern — the two Copy* messages are a straightforward miss, not an intentional shared label. Severity is genuinely low: it only fires when the 40-iteration infinity guard trips, the filePathDebug is included, and no numerical/memory behavior is affected — but the diagnostic will mis-name the marker family, which is exactly what the claim asserts. Suggested fix (per-method literal string naming its own marker: copyReplaceFill / copyReplaceAll) is straightforward and matches the DeleteThis/ChooseReplace convention.

**Suggested fix**

Make each message name its own marker (copyReplaceFill / copyReplaceAll) so an infinite-loop diagnostic points at the right marker type.

## Scanner notes

Scanned all 6 listed hand-written codegen-engine files in full and cross-checked against actual templates (Pivot.Operations.cs copyReplaceAll block; UnsafeBitsOP/Hash iProxy files combining alsoExpand[uint] with 4-value choose markers) and generated output (Ifloat/Idouble confirming the intentional interface-token expansion). Token substitution, choose/skipFor/alsoExpand parsers, and the Unity-vs-bootstrap paths are consistent; no leaks/use-after-dispose in these host files. The choose+alsoExpand interaction is safe because affected templates supply 4 pipe values matching the widened int/short/long/uint array. Malformed copyReplace/copyReplaceFill markers (missing end token) throw ArgumentOutOfRangeException before reaching the infinityGuard LogError - a hard build failure rather than a clean diagnostic, so loud and not reported as a defect.
