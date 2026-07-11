# Release doc/comment audit — consolidated summary (2026-07-11)

Eight parallel scans of the template source of truth (`CodeGen/TemplateSource*`), the markdown docs, and API coherence. Read-only; nothing was modified. Detail reports live next to this file.

## Reports

| Report | Scope | Findings |
|---|---|---|
| [doc-audit-op-a-to-l.md](doc-audit-op-a-to-l.md) | OP: Bidiag…LU (39 files) | 141 (HISTORY 91, TOO-LONG 43, JARGON 6, WRONG 1) |
| [doc-audit-op-m-to-r.md](doc-audit-op-m-to-r.md) | OP: MIP…Resample (35 files) | 58 (TOO-LONG 25, HISTORY 27, JARGON 4, NOISE 2) |
| [doc-audit-op-s-to-z.md](doc-audit-op-s-to-z.md) | OP: SVD…WindowType (31 files) | 16 (HISTORY 9, JARGON 3, TOO-LONG 2, WRONG 1, NOISE 1) |
| [doc-audit-core.md](doc-audit-core.md) | root + fProxy/iProxy/bool/Interfaces/Indices/Pivot/Realtime (49 files) | ~62 (HISTORY ~34, TOO-LONG ~19, WRONG 1, NOISE 1) |
| [doc-audit-sparse-modules.md](doc-audit-sparse-modules.md) | Sparse/Arena/Analysis/Statistics/ML/Hash/Debug (62 files) | ~72 (HISTORY 36, TOO-LONG 31, JARGON 4, WRONG 1) |
| [doc-audit-markdown.md](doc-audit-markdown.md) | README, CHANGELOG, docs/, package.json (56 files) | classification + fixes |
| [doc-audit-tests-benchmarks.md](doc-audit-tests-benchmarks.md) | test/benchmark templates + Tools (fast pass) | ~50 (mostly spec-doc refs) |
| [coherence-audit.md](coherence-audit.md) | API/naming/structure coherence, whole template surface | ranked findings |

Total: ~400 findings across ~216 library template files + docs. Only **4 factually wrong** comments found — the docs are accurate; the debt is process leakage and over-explanation.

## The systemic patterns (fix as sweeps, not file-by-file)

1. **Internal spec/ticket references in shipped comments** (~90+ hits, every area): `docs/draft-spec-*.md`, `docs/dev/rfc-memory-model.md §x.y`, ticket codes `R2/R3/R5/R6a/R8`, `OQ-7`, `P2/P3`, `STAGE n`, "Q4 ruling", "FM2/failure mode 1". One grep-driven sweep kills most of these.
2. **Optimization-campaign narration**: "an earlier version… measured 2x… REVERTED", A/B benchmark writeups with ms numbers and dates embedded in source (worst: `LP.fProxy.cs:325`, `UnsafeOP.Sparse.fProxy.cs:156`, `SparseOP.fProxy.cs:154`, `UnsafeOP.fProxy.cs`, `Consts.cs:45`).
3. **Agent/dev-workflow names leaked into source**: "see coder report" (`LP.DualSimplex.fProxy.cs:471`), "test-writer's SDA-vs-oracle check" (`Control.fProxy.cs:305`), "third-review finding" (`MIP.fProxy.cs:541`), a "fetched and read 2026-07-09" research note (`QP.fProxy.cs:762`).
4. **Essay-length public doc comments**: `LOBPCG.fProxy.cs:10` (74 lines), `QP.fProxy.cs:12` (70-line derivation), `Arena.cs:538` (~60 lines), `LstsqInfo` in `SolveInfo.cs:6` (~30 lines), `fProxyBSRBuilder.cs:19` (25-line bug diary).
5. **Copy-pasted arena-tracking doc debt** across all four type families (`fProxyN/MxN`, `iProxyN/MxN`, `boolN/MxN`): identical `Dispose()`/`AssertRecordAlive` history notes and RFC citations — clean once, apply to all six files.

## Factually WRONG comments (fix regardless of style pass)

- `SVD.fProxy.cs:177` — `thin()` allocation size claim omits the Ut/Vt Temp buffers.
- `FFT.Workspace.fProxy.cs:12` — twiddle table "~16 MB at N=1M" should be ~8 MB.
- `ML/KMeansEnums.cs:6` — k-means++ documented as O(k²·N·D); implementation is O(k·N·D).
- `Interfaces/LinearOperator.fProxy.cs:7` — public doc references the retired `Solvers` class (now Krylov/Blas/SVD/Eigen).

## Coherence: top items (see coherence-audit.md for full list)

1. **`fProxyComp` ships as a real public class** — `OP/UtilityOP.cs` class declaration sits outside the `//+copyReplace` block, so the raw proxy token survives codegen; `zeroInPlace` is stranded on a nonsense class instead of `floatComp`/`doubleComp`. This is a codegen bug, not just style.
2. **`ChooseMarkerDemo` ships as five public classes** in `Source/Generated/` — the demo file's own doc says it isn't public surface.
3. **Exception text says `QueryOP.<method>`** (~30 throws) but the public class is named `Query`.
4. **`Eigen.symmetric`/`valuesSymmetric` destroy `A`** without `InPlace` naming or `A_to_X` param rename — the one deviation from the settled destructive-naming rule.
5. **tolerance/maxIterations naming drifts four ways** across Krylov/Eigen/SVD/LOBPCG/QRCP — the solvers most likely to be used together.
6. **naming-style-guide.md documents a `_WS` workspace suffix that doesn't exist** — all 19 workspace structs are `*Cache`; the guide is stale.

## Markdown docs: top items (see doc-audit-markdown.md)

1. **LP/LAD stack (9 solver files) has zero public documentation** — absent from README, `docs/features/`, and CHANGELOG.
2. **CHANGELOG frozen at 0.1.0** while major features landed since.
3. Three `docs/features/*.md` files leak internals (ticket `OQ-7`, unit-test class names).
4. 15 spec/draft/research files sit loose at `docs/` root — move to `docs/dev/`.

## Suggested execution order

1. Codegen bug: `fProxyComp` (real API defect) + exclude `ChooseMarkerDemo` from generation.
2. The 4 WRONG comments.
3. Grep sweeps for patterns 1–3 (spec refs, campaign narration, agent names) — mechanical, high volume.
4. Condense the essay doc comments (pattern 4) + arena-family dedup (pattern 5).
5. Markdown: LP/LAD feature doc, CHANGELOG, docs/ reorganization.
6. Coherence judgment calls (Eigen destructive naming, tolerance-param unification) — API changes, decide before v1.0 freeze.
