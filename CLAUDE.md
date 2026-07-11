# LinearAlgebraBursted — project rules

## Source of truth

Templates under `Assets/LinearAlgebra/CodeGen/TemplateSource*` are the source of truth.
Never hand-edit `Assets/LinearAlgebra/Source/Generated` or `SourceTests` — edit the template
and regenerate (`Tools/regen.ps1`). Tests and compilation run headlessly via `Tools/*.ps1`.

## Comment policy (strict)

Code comments and XML docs state **contracts only**: what a member is for, what it destroys,
what it requires (SPD, allocator, sizes), what it returns. Short, plain words.

Everything else goes in the folder's `DEVLOG.md` (next to the template files), **never** in
code comments:

- development history ("an earlier version…", "changed from…")
- benchmark results, perf verdicts, rejected alternatives
- bug postmortems and debugging narration
- internal spec/ticket references (`docs/dev/*.md`, `R6a`, `OQ-7`, `STAGE n`)
- notes to reviewers or references to agents/workflow ("coder report", "third-review finding")

`DEVLOG.md` files live per template folder, are never processed by codegen (it only reads
`*.cs`), and never ship in the UPM package (package root is `Assets/LinearAlgebra/Source`).
Spec/ticket references are fine inside a DEVLOG.

DEVLOG entry format — under a `## <file or topic>` heading, newest first:

```markdown
## UnsafeOP.Sparse
- 2026-07-11 | Software prefetch on BSR spMV measured 8-56% slower, reverted. Don't retry. (was UnsafeOP.Sparse.fProxy.cs:156)
```

Date only (git has exact timestamps). Add `(was file:line)` when relocating an existing comment.

## Public docs

`README.md`, `CHANGELOG.md`, and `docs/features/*.md` are user-facing: short, concrete, no
dev history, no commit hashes, no internal spec/ticket references, no test-class names.
`docs/dev/` is internal and exempt.
