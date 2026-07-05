# Codegen & Rename-Refactor Lessons (LinearAlgebraBursted)

Hard-won rules from the pre-release naming/API-finalization sweep (2026-06-30/07-01). Terse on
purpose. Companion to `docs/naming-style-guide.md` (the conventions these mistakes were made
*trying to reach*) and `docs/perf-vectorization-lessons.md` (the same format, for kernel perf).

## PowerShell scripting traps
- **`-eq`/`-ne` are CASE-INSENSITIVE by default.** A rename script's `if ($text -ne $original)`
  change-detection will silently no-op on a file where every edit is a pure-case change (e.g.
  `SolveUpperTriangular` → `solveUpperTriangular`) — the comparison reports "no change" and the
  write is skipped. This is the single most consequential bug of the whole sweep: it silently
  skipped files whose ONLY edits were case-only, while files with at least one length-changing edit
  alongside case-only edits looked fine (masking the bug for a while). **Always use `-ceq`/`-cne`
  (case-sensitive) for change-detection in rename/rewrite scripts.**
- **Array-slicing an untyped `-split` result produces `Object[]`, not `String[]`.**
  `$newLines.AddRange($lines[0..$lastUsingIdx])` throws "cannot convert System.Object[] to
  IEnumerable[string]" against a `List[string]`. Fix: explicit `for` loop, cast each element to
  `[string]` before adding — don't rely on PowerShell to infer the element type through a slice.
- **Diagnose silent rename failures with an isolated repro**, not by staring at the script: dump
  `[object]::ReferenceEquals`, `.Length`, and a case-SENSITIVE `[string]::Equals(...,
  [StringComparison]::Ordinal)` on the specific before/after pair. That's what surfaced the `-ne`
  bug above — the symptom (file unchanged on disk) gave no hint the comparison itself was the bug.

## Codegen bootstrap catch-22
- **Unity won't run codegen unless the WHOLE PROJECT currently compiles** — not just the CodeGen
  assembly. Renaming a consumer (a benchmark, a hand-written test) to reference a class name that
  only exists in the NOT-YET-REGENERATED templates creates a chicken-and-egg deadlock: the rename
  breaks compilation, which blocks codegen, which is the only thing that could fix the compilation.
- **Fix pattern**: `git checkout <last-good-commit> -- <consumer-paths>` to temporarily revert just
  the consumers → run `regen.ps1` alone (this bootstraps `Generated/` from the new templates using
  the OLD, still-matching consumers) → delete any newly-orphaned generated files left over from the
  old names → re-run the rename script (it only touches the reverted consumer files this time,
  which now compile against the freshly regenerated types) → full `regen-and-test.ps1`.
- **Never blanket-delete `Generated/` to escape this.** It looks like the obvious "just start clean"
  move but deletes core data-struct types (`floatN`, `floatMxN`, ...) that hand-written
  non-generated code (`Source/Debug/PrintExport.cs`) needs just to COMPILE — this creates a DEEPER
  bootstrap trap than the one you were trying to escape, because now nothing in the project
  compiles, including the files codegen itself depends on transitively. Recovery required
  `git checkout <last-good-commit> -- Assets/LinearAlgebra/Source/Generated
  Assets/LinearAlgebra/SourceTests/Generated` to restore a known-good baseline before retrying
  incrementally. Use `Tools/prune-orphaned-generated.ps1` instead (targeted, keeps `.meta` files,
  wired as the first step of `regen.ps1`) — it deletes only files that are true orphans of the
  CURRENT templates, computed by mirroring `TemplateConverter.cs`'s own path logic, never a
  blanket wipe.
- **Codegen never deletes stale output on its own.** It overwrites at predictable paths but has no
  concept of "this template was renamed/moved/deleted, so its old output is now garbage" — the old
  file just sits there and causes CS0111 (duplicate class) or CS0103 (missing class), which ALSO
  blocks the bootstrap compile codegen needs to run. This is exactly what
  `prune-orphaned-generated.ps1` exists to close.

## Split vs merge (dropping a fProxy/iProxy prefix)
- **Generic type CONSTRAINTS don't participate in C# overload-signature uniqueness.**
  `L2<T>(in T a) where T : IUnsafefloatArray` and the double fragment's equivalent are IDENTICAL
  signatures once merged into one partial class — the `where` clause is invisible to the compiler
  for this purpose. Attempting to merge `Norms_OP` (drop the `fProxy` prefix) hit exactly this as
  CS0111 and had to be reverted to `fProxyNorms_OP`. See `docs/naming-style-guide.md`'s "Split vs
  merge safety" section for the full checklist — check BOTH criteria (arg-less/output-only methods,
  AND under-constrained generics) before attempting a merge, not just the first one.
- **A failed merge attempted mid-bootstrap-trap leaves stale generated files with the BROKEN bare
  name**, written before codegen could be re-run to fix them. Reverting the template alone isn't
  enough — the already-generated `.cs` output files (and everything that cross-references the now
  wrong name) need hand-patching or a fresh regen to actually unblock compilation. Prefer reverting
  BEFORE attempting a regen if a merge is at all uncertain — cheaper than patching stale generated
  output by hand afterward.

## Local edits with non-obvious blast radius
- **Introducing a new class name can shadow an existing local variable of the same name (CS0841 /
  forward-reference error).** Splitting `Ortho_OP` into `QR`/`LQ` classes broke test files that
  happened to declare `var QR = ...;` / `fProxyMxN LQ = ...;` as ordinary local variables in the
  SAME scope as a call to the new `QR.qrDecomposition(...)` / `LQ.lqDecomposition(...)` — C#'s
  block-scoping treats the local declaration as shadowing the type for the whole block, and a
  reference to the type BEFORE the local's declaration point is an error, not a fallback to the
  outer name. Fix: rename the locals (`QR` → `QRProduct`, `LQ` → `LQProduct`). When introducing a
  new top-level type name, grep the whole repo for that identifier used as a local/field/param
  before assuming the rename is done.
- **Adding a `using System;` can collide with `Unity.Mathematics.Random` (CS0104 ambiguity with
  `System.Random`).** Any file with a local `Random random = new Random(seed)` (common in this
  codebase's random-generation extension methods) breaks the moment `using System;` is added,
  because both namespaces now have a visible `Random` type and neither `using` wins by default.
  Cheaper fix than importing `System` and hoping for the best: fully-qualify just the few call
  sites that need it (`System.ArgumentException`, `System.ArgumentOutOfRangeException`) instead of
  adding the blanket `using`.
- **Changing an exception TYPE (not just wording) breaks existing `Assert.Throws<...>` tests that
  pinned the OLD type.** Standardizing on `ArgumentException`/`ArgumentOutOfRangeException` (away
  from bare `System.Exception`) is a real behavior improvement, but any test asserting
  `Assert.Throws<Exception>(...)` for that code path needs its generic argument updated to match —
  it will fail-not-pass once the thrown type narrows, even though the new behavior is strictly
  better. Grep for `Assert.Throws<Exception>` (the too-generic base type) as part of any exception
  sweep, not just `Assert.Throws<ArgumentException>`.
- **Dynamic exception messages (`who + ": " + detail`) are not Burst-safe.** Burst only reliably
  extracts a compile-time-CONSTANT string literal for the exception types it special-cases; runtime
  string concatenation defeats that. Shared validation helpers (`RequireXWorkspace(in ws, ...)`
  style) that took a caller-supplied `who` prefix string had to drop that parameter and use one
  static literal per call site instead.

## Block-comment / marker authoring traps
- **Never write a literal `*/` inside prose that itself lives inside a `/* ... */` doc comment**,
  even as a documentation EXAMPLE of a marker syntax that itself uses `/* */`. Describing the new
  `/*+choose[1e-6f|1e-14]*/1e-6f/*-choose*/` marker inside `GenUtils.cs`'s existing top-of-file
  `/* ... */` block comment closed that outer comment early at the first `*/`, spilling the rest of
  the intended-as-comment text into real code and cascading into CS1002/CS0116/CS1022/CS1031 errors
  far from the actual typo. Fix: reword the example to avoid an embedded `*/` (or use line comments
  for anything that needs to show `*/` literally).
- **A per-generated-type test needs the SAME per-type substitution mechanism for its OWN expected
  value, not a hardcoded literal.** `ChooseMarkerTests.fProxy.cs` originally asserted against
  `(fProxy)1e-6f` in both the float AND double generated test outputs — correct for float, silently
  WRONG for double (`1e-6f` cast to double ≠ the intended `1e-14`). Using the identical
  `/*+choose[1e-6f|1e-14]*/1e-6f/*-choose*/` marker in the test's own expected-value declaration
  fixed it — the test now resolves per-type exactly like the code under test does. General
  principle: whenever a template test's assertion needs a type-dependent value, drive it through
  the SAME codegen mechanism the implementation uses, don't hand-duplicate the per-type values.
- **Never write a marker's own literal start token as PROSE inside a doc comment describing it** —
  not just the `choose`/`*/` case above; this generalizes to every marker (`//+skipFor[...]`,
  `//+copyReplace`, `//alsoExpand[...]`). The converter's marker parsers do a plain
  content-sensitive `string.IndexOf` over the WHOLE file text — they have no idea what a real C#
  comment boundary is — so a sentence like "see the `//+skipFor[u]` block below" is itself a second,
  PHANTOM marker occurrence. Concretely: writing `//+skipFor[u]` in a header comment, then a real
  `//+skipFor[u] ... //-skipFor` block further down, makes `SkipForReplace` treat the header's
  bracket-less/malformed text as the marker instance, silently mis-parse or consume the real block's
  own closing marker, and corrupt everything downstream — this surfaced as CS0111 in one file and, in
  a second file, as the file's tail getting silently chewed away (missing braces →
  CS0106/CS1022 dozens of lines later, nowhere near the actual typo). Same failure mode hit
  `//+copyReplace` when a doc comment said "the inner iProxy `//+copyReplace` block" — `CopyReplace`
  found the phantom start, then hunted for `//-copyReplace` and grabbed the wrong span, throwing
  `ArgumentOutOfRangeException: length ('-2250')` deep in `TemplateConverter.CopyReplace`. Fix:
  describe markers by name in prose ("a skipFor-marked block", "a copy-replace block") without ever
  reproducing the literal `//+`/`//-`/`/*+`/bracket token sequence; grep the whole diff for each
  marker's exact start token after writing ANY comment that discusses codegen mechanics, not just
  once at the end.
- **A proxy-token SUBSTRING anywhere in a file's CONTENT — even inside a plain comment — changes how
  the whole file is classified and emitted, if the FILENAME carries no proxy token.** The converter
  (`TemplateConverter.cs:37-39`) treats a file as singular only when neither its filename NOR its
  content contains a proxy token; a non-singular file whose filename lacks the token still gets routed
  to multi-emission with the family chosen purely by content (`:64-82`), and the per-type output path
  comes from `relativePath.Replace(proxy, typeStr)` (`:88`) — a NO-OP when the filename has no token.
  Result: all per-type emissions collapse onto ONE output path, last-write-wins, with whatever the
  final iteration substituted (observed as `"longN/fProxyN.Dispose()"` garbage in generated output
  during the bool records migration). So in any token-free-NAMED file (`boolRecords.bool.cs`,
  `Arena.cs` sans its `//singularFile//`…) never write a literal proxy name even in prose — reference
  concrete safe names (`floatN`, `intN`) instead, and keep `//singularFile//` markers intact. Grep new
  token-free-named files for proxy substrings before regen.
- **A brand-new codegen-generated CONCRETE type name (e.g. `uintN`, the product of substituting a
  proxy token) does not exist as a real type anywhere in `TemplateSource`'s own raw, unprocessed
  compile** — only the proxy token itself (`iProxyN`, wired up as a real placeholder struct in
  `proxyStructs.cs`) is a valid stand-in there. Writing a hand-added member that references the
  concrete name literally (`public uintN uintVec(...) => ...;`) compiles fine in the FINAL generated
  output (where `uintN` is real) but fails TemplateSource's own compile with "type or namespace name
  'uintN' could not be found" — because that pass never substitutes anything, it just compiles the
  literal template text as-is. Route any new member through the proxy token inside a (possibly
  newly widened) `//+copyReplace`/`//alsoExpand` block instead of hand-writing the concrete name;
  see `iProxyN.Shortcuts.cs`'s uint cross-shortcut, which had to move from a hardcoded `uintN`
  method to widening the existing `iProxyN`-token copyReplace block.

## Misc
- **A `.iProxy.cs`/`.fProxy.cs` file's routing is controlled by its FILENAME suffix, not its
  folder** — but a misrouted folder (an `.iProxy.cs` file physically sitting under a `fProxy/`
  test folder) still generates output at a folder path that mirrors the SOURCE folder, landing
  generated int/short/long files under `Generated/fProxy/` instead of `Generated/int/` etc. `git mv`
  to the correct folder, then delete the stale orphans at the old generated path (another case
  `prune-orphaned-generated.ps1` now catches automatically).
- **`//+deleteThis` blocks must still be valid, compilable code** in `TemplateSource`'s own
  standalone assembly compile, even though they're stripped from generated output — don't assume
  a block is dead code just because it's excluded from what ships. See `Consts.cs`'s
  `fProxyZeroThreshold`/`fProxyEpsilon`/`fProxySqrtEps` (other template files reference these by
  name at raw-template-compile time).
