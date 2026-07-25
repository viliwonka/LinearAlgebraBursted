# Agent brief — domain-exposition bloat in comments and XML docs

Read-only scan. Report findings; do not edit.

## What you are looking for

Comments and XML docs that **teach the reader the subject matter** instead of stating the
**contract of the code**. The reader of `LP.solve` already knows what a linear program is — they
went looking for an LP solver. Restating the definition wastes their time and ours.

This is NOT the jargon sweep (already done: "Burst-monomorphized static-dispatch shape" etc.) and NOT
the history sweep (already done: "an earlier version…", benchmark logs → DEVLOG). Those found
*wrong-register* prose. You are looking for prose that is correct, well-written, and **unnecessary**.

## The test — apply it to every sentence

> Does a competent caller need this sentence to call the function CORRECTLY?

- **Needs it → KEEP.** Shapes, lengths, units, which argument is overwritten, what is destroyed,
  what must be SPD/sorted/distinct, allocator and job-safety, error/status semantics, non-obvious
  preconditions ("variables are non-negative; split a free variable into a +/- pair"), and which
  overload to pick when the choice is not obvious from the name.
- **Doesn't need it → CUT.** Definitions of the problem class, canonical mathematical formulations
  that restate the signature, derivations of why the method works, textbook motivation, tutorials on
  how to model something in the paradigm.

When unsure, KEEP. Deleting a real precondition is far worse than leaving a redundant sentence, and
this codebase has had bugs whose only warning lived in a comment.

## Calibration — real examples from `OP/LP.fProxy.cs`

CUT — restates the signature as mathematics; anyone calling an LP solver knows this shape:
```
//     minimize    cᵀx
//     subject to  Aᵢ·x  {≤, =, ≥}  bᵢ    (per-row sense in `senses`)
//                 x ≥ 0
```

CUT — a tutorial on LP modelling. The *cross-reference* to `lad` is worth one clause; the derivation
is not:
```
// L1 regression (least absolute deviation) is the flagship application: minimize ‖Ax − b‖₁ over a
// FREE x is exactly an LP once each residual is split into a +/− pair (see `lad`).
```

CUT — a benchmark verdict in a PUBLIC XML doc. Already forbidden by CLAUDE.md (measurements belong in
DEVLOG); it also rots the moment anything is re-benchmarked:
```
/// <param name="method">Backend (default RevisedSimplex — fastest exact backend at every
/// benchmarked size on cold solves and the fastest infeasibility certifier (1-2 pivots); …
```
The keepable residue is "default RevisedSimplex; pick DualSimplex for warm re-solves,
InteriorPoint for very ill-conditioned vertices" — routing guidance a caller acts on, with the
justification dropped.

KEEP — a real precondition the caller cannot infer:
```
/// Variables are non-negative; model a free variable by splitting it into a +/− pair, or use
/// <see cref="lad"/> which does that for you.
```

KEEP — a real lifecycle contract:
```
// Job-safe: the cores allocate their scratch from Allocator.Temp and dispose it before returning.
```

## Scope

Scan in this order (heaviest expected return first):

1. `Assets/LinearAlgebra/CodeGen/TemplateSource/**` — the source of truth. **Only report findings
   here**; `Source/`, `SourceTests/Generated/`, `Benchmarks/Generated/` are codegen OUTPUT and must
   never be edited. If a finding's generated twin appears, report the TEMPLATE path only.
2. `Assets/LinearAlgebra/Benchmarks/*.cs` and `Assets/Demos/**` — hand-written, in scope.
3. `docs/features/*.md` — user-facing; same test applies. `docs/dev/` and `docs/audit/` are internal
   and EXEMPT.

Expect the worst offenders in the big feature entry points — `OP/LP*.cs`, `OP/QP*.cs`, `OP/MIP*.cs`,
`OP/Krylov*.cs`, `OP/Eigen*.cs`, `OP/SVD*.cs`, `ML/*.cs`, `Control/*.cs` — and specifically in
**file-level header blocks** and **public XML `<summary>`** rather than in inline code comments,
which in this codebase are already terse.

## Report format

One markdown table, ordered by lines-saved descending:

| file:line | category | lines | verdict |

`category` ∈ {`problem-definition`, `math-restates-signature`, `derivation`, `modelling-tutorial`,
`benchmark-verdict-in-public-doc`, `motivation`}.

For each finding give the offending text quoted, and — where anything in it is worth keeping — the
one-sentence residue that should survive. Do not propose rewrites beyond that residue.

End with a total line count and the three files worth doing first.

## Do not

- Do not edit anything. Report only.
- Do not touch `Source/`, `SourceTests/Generated/`, `Benchmarks/Generated/` — regenerated from
  templates; edits there are lost and cause drift.
- Do not flag `<param>`/`<returns>` tags for existing merely because they are terse.
- Do not flag DEVLOG.md files — history belongs there by design.
- Do not flag a sentence solely for being long; length is not the criterion, necessity is.
