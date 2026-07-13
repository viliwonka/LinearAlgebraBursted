# Release-readiness scan briefs — 2026-07-13

Orchestration: 8 **wide** scans (one dimension each, across all templates) + ~10 **narrow**
scans (one folder partition each, all dimensions, read every line). Model routing: `sonnet`
for the comments/docs dimensions (W1, W8), default model for everything code-logical.
Findings go to `docs/audit/release-scan-2026-07-13/`.

---

## Shared preamble (paste at the top of EVERY agent prompt)

```
You are one scanner in a release-readiness audit of a Unity Burst linear algebra library.

SCOPE — templates only:
  Assets/LinearAlgebra/CodeGen/TemplateSource/**        (production templates, 236 .cs)
  Assets/LinearAlgebra/CodeGen/TemplateSourceTests/**    (test templates)
  Assets/LinearAlgebra/CodeGen/TemplateSourceBenchmarks/** (benchmark templates)
NEVER report findings against Assets/LinearAlgebra/Source, SourceTests/Generated, or
Benchmarks/Generated — those are 100% codegen OUTPUT. If you spot a problem there, trace
it back to the template and report the TEMPLATE file:line.

CODEGEN CONTEXT: templates use proxy tokens (fProxy → float/double; iProxy → integer
types; also Comp/vector/matrix name tokens) and `//+choose[...]` blocks that gate code
per generated type. Before scanning, skim Assets/LinearAlgebra/CodeGen/TemplateConverter.cs
to learn the exact token/marker rules. A line can be correct for float and WRONG after
substitution for double/int/uint — always reason about every generated variant.

RULES:
- Read-only. Do not edit anything, do not run the test suite, do not regen.
- Before flagging a "leftover" or "odd" decision, check the folder's DEVLOG.md — history
  and rejected alternatives are ALLOWED there (and only there).
- Calibrate: this library has already survived several audit passes. Report real defects
  and genuine release-blockers, not taste. If a whole area is clean, say so in one line.

OUTPUT: write your report to docs/audit/release-scan-2026-07-13/<your-report-name>.md.
For each finding:
  - Severity: HIGH (wrong result / crash / API lie) | MEDIUM (inconsistency, misleading
    doc, missing guard) | LOW (polish)
  - Template file:line
  - One-sentence defect + the exact quoted line or a concrete failure scenario
    (inputs/type-variant → wrong outcome)
  - One-line suggested fix direction (do NOT apply it)
End with a summary table: findings per severity, areas confirmed clean.
Your final message: report path + counts + the HIGH findings inline.
```

---

## Wide scans — one dimension each

### W1 — Comments & XML documentation (model: **sonnet**)

```
DIMENSION: comments and XML docs, templates only. Report: report-W1-comments.md

Policy (CLAUDE.md, strict): code comments and XML docs state CONTRACTS ONLY — what a
member is for, what it destroys, what it requires (SPD, allocator, sizes), what it
returns. Short, plain words. Everything else belongs in the folder's DEVLOG.md.

First run Tools/check-doc-leaks.ps1 and include its output. Then sweep every template
for what the script can't catch:

FLAG (must move to DEVLOG.md or be deleted):
- historical context: "previously", "used to", "was changed", "an earlier version",
  "now uses", "reverted", version/date references
- dev-speak: reviewer/agent references, "per the audit", "coder report", ticket/spec
  refs (R6a, OQ-7, STAGE n, docs/dev/*.md)
- benchmark numbers, perf verdicts, rejected alternatives in comments
- narration ("// increment counter"), commented-out code, TODO/FIXME/HACK
- justification-to-reviewer comments ("this is correct because...")
For each, propose the one-line DEVLOG entry (## <file> / - 2026-07-13 | ... (was file:line)).

ALSO FLAG (doc bugs):
- XML docs that LIE: wrong param described, stale contract, destructive method whose doc
  doesn't say what it destroys, doc claiming a requirement the code doesn't check or
  vice versa
- copy-paste doc errors (vector op described as matrix op, wrong method name in <summary>)
- public API members missing docs entirely where siblings have them
- docs that won't make sense after token substitution (e.g. mention "float" literally in
  a template that also generates double)
```

### W2 — Error handling & exceptions (default model)

```
DIMENSION: error handling, templates only. Report: report-W2-errors.md

First learn the house patterns: solvers return diagnostic structs with shared enums
(IterativeSolveStatus / DirectSolveStatus) instead of throwing on numerical failure;
argument/size validation conventions; what's allowed under Burst (no managed allocs in
exception paths inside jobs).

FLAG:
- numerical failure paths that throw where siblings return a status (or vice versa)
- missing size/shape/allocator validation on public entry points where siblings have it;
  redundant validation inside hot inner loops
- exception messages that are wrong: name a different method, swap rows/cols, or
  interpolate values in a way Burst can't compile
- silently-swallowed failure (status computed but never set, early return losing an
  error), unreachable guards
- inconsistent guard style for the same condition across sibling files
- test templates that assert on exception TYPES/messages that the API no longer throws
```

### W3 — Numerical correctness (default model)

```
DIMENSION: numerical errors, templates only. Report: report-W3-numerics.md

FLAG:
- unguarded division / normalization by a possibly-zero quantity (norms, pivots, dot
  products); sqrt of a possibly-negative value from cancellation
- exact float equality (== 0f, == 1f) where a tolerance is needed — but check siblings;
  some exact checks are intentional (e.g. structural zeros)
- tolerance/epsilon literals wrong for the precision: a 1e-6 hardcoded in a template
  that also generates double (should scale per type), or eps chosen for double leaking
  into float
- overflow in integer variants: index math (row*cols), accumulators, abs(int.MinValue),
  uint subtraction underflow
- catastrophic cancellation patterns with a known stable alternative (hypot-style,
  two-pass variance, Kahan where the contract implies it)
- loss of symmetry: SPD/symmetric updates writing only one triangle when the contract
  says both, or reading the wrong triangle (the library recently standardized on lower)
- convergence checks comparing against the wrong norm/scale (absolute vs relative)
```

### W4 — Type-split correctness / `//+choose` gating (default model)

```
DIMENSION: correct per-type splits, templates only. Report: report-W4-typesplit.md

Read TemplateConverter.cs first to learn exactly which types each proxy expands to and
how //+choose[...] gates code. Then audit EVERY //+choose block and every fProxy/iProxy
file split.

PRIMARY FOCUS — per-type constants and literals. For every numeric literal and named
constant in a template, ask: is this value right for EVERY type the template generates?
- epsilon/tolerance constants sized for float (1e-6, 1e-7) that also generate into the
  double variant, where a double-appropriate value (~1e-12 to 1e-15 scale) is expected —
  and the reverse (double-scale eps generated into float, where it underflows the
  precision and the check never fires)
- machine-epsilon-derived thresholds that should be expressed per-type (via the proxy's
  eps token or a //+choose split) but are hardcoded to one precision
- suffix/limit constants that don't substitute: a literal `f` suffix, float.MaxValue /
  float.Epsilon / MathF-style calls surviving into the double variant, or unsuffixed
  double literals silently narrowing in float
- convergence iteration counts or safeguards tuned to one precision and shared by both
- integer variants with real-only literals (0.5, 1e-6) or math that truncates silently
- uint variants of code that subtracts or negates

ALSO FLAG:
- inherently-real ops (norms with sqrt, solvers, eigen, anything with fractional
  intermediate values) leaking into int/uint variants. Policy: inherently-ℝ ops are
  float-only — but do NOT recommend deletion; record each leak as an OPEN QUESTION
  for the maintainer to rule on, with the op, the leaking type, and what the generated
  code currently does (truncates? compiles at all?)
- asymmetric coverage that looks accidental: float has it, double doesn't (or reverse),
  with no DEVLOG/policy justification
- //+choose blocks whose branches drifted apart: a bug fixed in the float branch but
  not the double branch, or vice versa
- TESTS gated to a different type set than the API they test (API generates double but
  test only covers float, or test would compile against a variant that doesn't exist)
- proxy tokens used inconsistently in one file (hardcoded 'float' where fProxy was meant)
```

### W5 — Logic errors (default model)

```
DIMENSION: logic bugs, templates only. Report: report-W5-logic.md

The library is ROW-MAJOR (M_Rows × N_Cols). Hunt:
- indexing: row/col swapped, i*N_Cols+j vs i*M_Rows+j confusion, off-by-one loop
  bounds, wrong bound after a transpose or on non-square matrices (test mentally with
  M≠N — square matrices hide these)
- copy-paste divergence between near-identical siblings: upper vs lower triangular,
  row vs column variants, matrix vs vector overloads — diff them mentally
- in-place aliasing: dest overlapping src where the algorithm reads stale positions
- pivot/permutation bookkeeping applied in the wrong order or direction
- inverted or short-circuited early-exit conditions; loops that can exit with
  uninitialized outputs
- wrong variable used after a rename-shaped edit (a and b both in scope, wrong one read)
- benchmark templates timing the wrong thing (setup inside the timed region, dead-code
  elimination of the result)
```

### W6 — Naming, semantics & method names (default model)

```
DIMENSION: naming and semantics, templates only. Report: report-W6-naming.md

Canon: docs/naming-style-guide.md, plus settled decisions:
- solver grid: decomp / decompInPlace / decompSolve / solveInPlace; destructive
  one-shots; A_to_Q / b_to_x style param names for destroyed-and-repurposed buffers
- InPlace suffix appears exactly when the method destroys/overwrites its input —
  flag any method that mutates without the suffix, or has the suffix but doesn't
- purged tokens must not resurface: Elem (→Comp), Linear (→Blas), BSM (→BSR), _OP
  suffix on non-data types; renamed: symmetricInPlace, maxIterations, tolerance
- M_Rows / N_Cols are KEPT — do not flag them
FLAG:
- method names that misdescribe (a 'solve' that also factorizes and destroys A without
  saying so; a 'get' that allocates; a predicate not reading as bool)
- parameter names that lie about direction/role (in vs out vs scratch)
- same concept named differently across sibling files, or same name meaning two things
- public surface accidents: members public that siblings keep internal
Report inconsistencies with BOTH locations so the fix direction is clear.
```

### W7 — Style & consistency (agent: **Creative design agent**)

```
DIMENSION: style, code smell, sore thumbs — templates only. Report: report-W7-style.md

This is the "would a stranger reviewing the released source raise an eyebrow" pass:
- dead code, unused parameters/locals, duplicated helper blocks that siblings share
  via a common utility
- formatting outliers: one file with different brace/spacing/member-ordering than the
  rest of its folder
- oddball leftovers: debug prints, magic numbers with no contract meaning, weirdly
  named locals (tmp2, foo), stray regions
- API sore thumbs: an overload ladder inconsistent with the iterative-solver ladder
  used elsewhere (NOTE: the existing solver overload COUNT is settled — don't propose
  trimming arity), a struct field order that differs from its twin
Calibrate hard: taste-only nitpicks are LOW; only consistency breaks visible in the
generated public package are MEDIUM+.
```

### W8 — Public docs (model: **sonnet**)

```
DIMENSION: public-facing markdown. Report: report-W8-public-docs.md
Scope: README.md, CHANGELOG.md, docs/features/*.md, Third Party Notices.md.
(docs/dev/** and DEVLOG.md files are internal — out of scope.)

IMPORTANT: report findings ONLY. Do NOT rewrite or draft prose — the maintainer
hand-writes all public prose. Findings are facts, not suggested wording.

FLAG:
- factual drift vs the templates: documented method/param names that don't exist or
  were renamed, wrong namespaces, wrong contracts (says non-destructive, is
  destructive), stale type lists (claims double support that isn't generated, or
  misses support that is)
- code examples that would not compile against the current API
- forbidden content: dev history, commit hashes, internal spec/ticket refs
  (docs/dev/*.md, R6a, OQ-7, STAGE n), test-class names, benchmark methodology
  narration
- coverage gaps: a shipped user-facing feature absent from README/features docs;
  CHANGELOG missing recent behavior changes (there are at least 2 known pending)
- broken relative links, wrong file paths, package name/version inconsistencies
  between README, CHANGELOG, and Source/package.json
```

---

## Narrow scans — per-folder deep pass (default model; run AFTER wide, one agent per partition)

Partition (≈15–30 files each; OP split by subject):

| # | Partition |
|---|-----------|
| N1–N5 | `TemplateSource/OP` split into 5 alphabetical chunks (~23 files each; agent lists the folder, sorts, takes its chunk) |
| N6 | `Sparse` (20) |
| N7 | `Arena` (18) + root `TemplateSource/*.cs` (11) |
| N8 | `fProxy` + `iProxy` + `bool` (28) |
| N9 | `Debug`, `Interfaces`, `Hash`, `Pivot`, `Indices`, `Realtime` (25) |
| N10 | `ML`, `Statistics`, `Analysis` (18) |
| N11–N12 | `TemplateSourceTests` split in 2 |
| N13 | `TemplateSourceBenchmarks` |

```
NARROW SCAN of partition: <files/folders>. Report: report-N<k>-<name>.md

Read EVERY line of every file in your partition — this is depth, not sampling. Apply
ALL dimensions at once: comment policy (contracts only — no history/dev-speak/perf
verdicts), error handling consistency, numerical safety, //+choose type-split
correctness for every generated variant, logic (row-major indexing, aliasing,
copy-paste divergence between siblings), naming canon (docs/naming-style-guide.md;
M_Rows/N_Cols are settled — keep), and style outliers.

Extra duties unique to the narrow pass:
- diff sibling files (fProxy vs iProxy versions, upper vs lower, row vs col) line by
  line and flag drift that looks accidental
- verify every XML doc contract against the code below it
- check the folder's DEVLOG.md exists if the folder has non-obvious decisions, and
  that no code comment duplicates what DEVLOG already records
```

---

## Narrow-pass addendum — recurring patterns the wide pass surfaced (2026-07-13)

Every narrow agent must ALSO sweep its partition for these confirmed wide-pass patterns:

1. **Role-swapped InPlace wrappers** (wide HIGH, `mulInPlace`): for every extension
   method forwarding to an Unsafe kernel, verify the receiver/argument pointer roles
   against the kernel's actual parameter semantics — which operand is mutated?
2. **Rename stragglers**: `maxIter` vs `maxIterations`, `tol`/`relTol` vs
   `tolerance`/`relativeTolerance`, retired names in docs/exception messages
   (`MatrixMetrics`, `StatsOP`, `BSM`, `Solvers`, `_OP`).
3. **Missing `InPlace` suffix** on a method whose doc/behavior destroys an input
   (wide HIGH: `Eigen.valuesQR`).
4. **`[NoAlias]` violations**: the same pointer passed to two `[NoAlias]` parameters.
5. **Sibling-validation gaps**: an overload missing a size/shape check all its
   siblings perform (wide: `Blas.dotRows`).
6. **Literal type keywords surviving substitution**: `float`/`0.3f`-style tokens in
   fProxy templates that generate verbatim into the double variant.
7. **Test-template comment-policy debt** (N11–N13 especially): ticket codes
   (R6a/FM2/STAGE n), inline bug postmortems, "per the spec", measured baselines —
   propose the DEVLOG.md relocation for each.

## Launch notes

- Wide pass: W1 + W8 with `model: sonnet` (W1/W8 are text; the maintainer wants Sonnet's
  plainer register there). W7 on the Creative design agent. W2–W6 on code-review agents /
  default model. All 8 can run concurrently.
- Narrow pass after reading wide reports (wide findings tell narrow agents what patterns
  to also sweep for — append any recurring wide finding as an extra bullet).
- Nobody fixes anything; fixes are a separate triaged pass after human review of
  `docs/audit/release-scan-2026-07-13/`.
