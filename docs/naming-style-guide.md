# Naming & Style Guide (LinearAlgebraBursted)

Reference for keeping new code consistent with the rest of the library — written for a linter/
reviewer agent to check changes against, not just for humans. Terse on purpose. Companion to
`docs/codegen-refactor-lessons.md` (mistakes/pitfalls hit while building this) and
`docs/perf-vectorization-lessons.md` (Burst vectorization rules).

## Method naming
- **camelCase**, with one sub-rule: a **leading** acronym is lowercased (`svdThin`, `luDecomposition`,
  `qrDecomposition`, `fft`) because camelCase needs a lowercase first letter; a **trailing/mid**
  acronym stays LOUD (`eigenvaluesQR`, `normalizeL2`).
- In-place suffix is **`Inpl`**, not `Inplace` (e.g. `mulInpl`, not `mulInplace`).
- Predicates are **lowercase camelCase** (`isSymmetric`, `isDiagonal`, `whichTrue`) — NOT Pascal
  `Is...`. Confirmed against Unity.Mathematics' own convention (`math.isnan`, `math.isfinite`,
  `math.isinf` — always lowercase, never `IsNan`).
- Exception/validation messages: `"MethodName: what went wrong"`, one static string literal — see
  Exceptions below for why "static" is load-bearing, not just style.

## Type naming
- Acronyms in **type names** are LOUD: `SVD`, `QR`, `LQ`, `FFT`, `LU` (contrast with the
  leading-acronym-lowercase rule for methods above — these are different rules for different things).
- **`_OP` suffix** = a stateless bag of free functions over buffers (`Stats_OP`, `Norms_OP`,
  `Elem_OP`) — a category marker, paired with a semantic prefix describing *what* it bags. A class
  named just `_OP` with no semantic prefix (the historical `fProxy_OP`) is a smell — split it or
  name it for its content (this project did both: split into `Linear_OP` + `Elem_OP`).
- **No suffix** = a named factorization/algorithm (`LU`, `SVD`, `Cholesky`, `Eigen`, `Bidiag`, `QR`,
  `LQ`, `Solvers`) — the algorithm's own name is the description, no `_OP` needed.
- **`_WS` suffix** = a reusable zero-alloc workspace struct (`fProxyBidiag_WS`, `fProxySVDThin_WS`).
- **A class name containing `fProxy`/`iProxy`** is SPLIT — codegen generates a *separate* class per
  concrete type (`floatFoo`, `doubleFoo`, ...). A **bare name** (no proxy token) is MERGED — codegen
  emits the *same* class name once per type, and C# partial-class merging combines them into ONE
  logical type with per-type method overloads. See "Split vs merge safety" below before dropping a
  prefix — getting this wrong produces a compile error, not a silent bug, but it's a two-step trap
  (see codegen-refactor-lessons.md's Norms_OP entry).

## Split vs merge safety (checklist before dropping a fProxy/iProxy prefix)
A class is safe to merge (drop its prefix) ONLY if **both** hold for every method:
1. **No arg-less / output-only method.** A method whose ONLY type-identifying information is its
   RETURN type (e.g. a hypothetical `identity(int n)` factory) can't have its precision inferred at
   a `var x = Foo.identity(3);` call site once merged, and two such methods (one per type) collide
   as CS0111 (Arena already owns all factory-style methods in this library specifically so this
   never comes up in practice — but check for it).
2. **No generic `<T>` method lacking a non-generic, precision-bearing parameter.** C# does NOT count
   generic type-CONSTRAINTS toward method-signature uniqueness — only concrete (non-generic)
   parameter types do. `L2<T>(in T a) where T : unmanaged, IUnsafefloatArray` and the `double`
   fragment's `L2<T>(in T a) where T : unmanaged, IUnsafedoubleArray` are IDENTICAL signatures once
   merged (the constraint is invisible to overload resolution) → CS0111. This is why `Norms_OP`
   stays split (`fProxyNorms_OP`) while `Linear_OP` (all concrete `fProxyN`/`fProxyMxN` params, no
   bare generics) safely merged, including across the fProxy/iProxy boundary (float/double/int/
   short/long dot/trans all coexist as overloads in one `Linear_OP` class).
- If both hold, merging is safe **today** — but note it's only a *default* choice, not a spec
  guarantee: a future arg-less factory method added to a currently-safe-to-merge class would force
  re-splitting it. Utility bags in the Stats/Norms mold default to staying split as a hedge against
  that; permanent-by-construction primitive bags (`Linear_OP`, the `LU`/`SVD`/`QR`/`LQ`/`Cholesky`/
  `Eigen`/`Bidiag`/`Solvers` factorization family) merge freely since Arena already owns every
  factory-style method in the library, so this category of class will never need one.

## Arena / ArenaExtensions
- Abbreviated **`Vec`/`Mat`**, not `Vector`/`Matrix` (`fProxyVec`, `fProxyMat`, `fProxyRandomVec`,
  `fProxyIdentityMat`) — chosen over the spelled-out form purely for lower total churn (the core
  `Arena` allocators are the highest-traffic names in the whole library); Unity.Mathematics itself
  doesn't have a directly-applicable precedent here since it embeds concrete sizes (`float3`) rather
  than using a generic Vec/Mat word at all.
- `Arena` (the struct) holds only the core bump-allocator primitives (`fProxyVec`, `fProxyMat`,
  `Indices`). Everything else (galleries, random constructors, FFT reductions, queries, per-algorithm
  workspace factories) is a `this ref Arena arena` extension method on the static
  `ArenaExtensions` class — keeps `Arena` itself a lean core allocator.
- No `DB_`-style debug prefix on introspection helpers (`isPersistent`/`isTemp`, not
  `DB_isPersistent`) — plain lowercase predicate naming applies uniformly, no special marking for
  debug-only members.

## Namespaces
- `LinearAlgebra` — the public surface.
- `LinearAlgebra.Internal` — raw-pointer kernel classes (`Unsafe_OP`, `UnsafeBool_OP`,
  `UnsafeSelect_OP`). **Public, not `internal` visibility** — namespace-signaled as
  advanced/internal-style (mirrors `System.Runtime.CompilerServices.Unsafe`), not access-restricted.
  Consumers need `using LinearAlgebra.Internal;`.
- Domain sub-namespaces (`LinearAlgebra.Stats`, `LinearAlgebra.ML`, `LinearAlgebra.Gallery`,
  `LinearAlgebra.Realtime`) group a feature area; PascalCase, singular.

## Workspace-overload pattern
Every decomposition/algorithm that allocates scratch should offer BOTH:
1. A **zero-alloc primitive**: `Method(..., ref fProxyXxx_WS ws)`, validated by a private
   `RequireXxxWorkspace(in ws, ...)` helper that throws `ArgumentException` (static message) if any
   field is mis-sized.
2. An **allocating convenience wrapper**: same name, fewer params, allocates its own
   `Allocator.Temp` scratch and delegates to the primitive (or to a workspace-free reference
   algorithm — whichever exists first; the allocating overload was usually written first
   historically, then the workspace overload added later without changing the allocating one's
   observable behavior).
- Workspace structs **nest** other workspace structs when the algorithm internally calls another
  workspace-capable algorithm (`fProxySVDThin_WS` nests `fProxyBidiag_WS`; `fProxyLQMinNormSolve_WS`
  nests `fProxyLQ_WS`) — this is how "zero-alloc all the way down" is achieved without duplicating
  buffers. Always prefer nesting the dependency's own `_WS` type over duplicating its fields.
- `Arena.fProxyXxx_WS(this ref Arena arena, ...)` factory methods live in `ArenaExtensions`,
  alongside everything else.

## Exceptions
- **`ArgumentException`** for shape/dimension mismatches ("must be square", "must equal N_Cols").
- **`ArgumentOutOfRangeException`** for index/bounds violations ("index out of bounds", "must be
  bounded inside vector dimensions").
- **No bare `System.Exception`.**
- **No custom exception types.** Burst only reliably supports `throw new <built-in-type>("static
  string literal")` — it extracts the message at compile time for the small set of exception types
  it special-cases; a custom subclass is a managed reference type outside that supported subset and
  risks silently failing (or just not compiling) the moment a consumer wraps a call in their own
  `[BurstCompile]` job, which is this library's core use case. Unity.Mathematics itself defines zero
  custom exceptions either, for the same reason.
- **Exception messages MUST be static string literals — no runtime concatenation.** `who + ":
  message"` (a caller-supplied prefix) is NOT Burst-safe; if several call sites need to be
  distinguished, either give each call site its own literal message, or drop the distinguishing
  prefix (usually fine — the message already says *what* went wrong even without saying *which*
  overload triggered it).

## Codegen markers (see `Assets/LinearAlgebra/CodeGen/GenUtils.cs` for the authoritative list)
- `//singularFile//` — this file is NOT multiplied per type; single output, same path.
- `//+copyReplace ... //-copyReplace` — duplicate a code region once per type in the file's type
  family (float+double, or int+short+long).
- `//+copyReplaceAll ... //-copyReplaceAll` — same, but ALL types (float+double+int+short+long+bool).
- `//+copyReplaceFill[sep] ... //-copyReplaceFill` — same as copyReplace, joined by `sep` between
  copies.
- `//+deleteThis ... //-deleteThis` — strip a block from generated output. The block must still be
  **valid, compilable code in TemplateSource's own standalone compile** (TemplateSource is a real,
  separately-compiled assembly — "not meant to be used directly" per its own doc comment, but it
  still has to compile). Don't assume a `//+deleteThis` block is inert; if it's expected to be
  referenced elsewhere in the SAME file (e.g. a constant used by other members in that file), it
  needs to exist as a real member even though it's stripped from the generated output — see
  `Consts.cs`'s `fProxyZeroThreshold`/`fProxyEpsilon`/`fProxySqrtEps` for a concrete example.
- `/*+choose[v0|v1|...]*/placeholder/*-choose*/` — inline per-generated-type literal substitution
  (added 2026-07-01): resolves to `v0` for the first type in the file's type family, `v1` for the
  second, etc. **Block-commented** (`/* */`), not line-commented (`//`), specifically so it can sit
  mid-statement without swallowing the rest of the line. Never embed a literal `*/` inside a prose
  example of this marker written INSIDE a `/* */` doc comment — it prematurely closes the outer
  comment (see codegen-refactor-lessons.md).

## Testing conventions
- A workspace-overload's test suite verifies **equivalence** to the allocating overload (same
  inputs → same outputs within a small precision tolerance), reuse-across-calls (no stale state),
  mis-sized-workspace throw guards, and `Arena` factory field-sizing — see any `*WorkspaceTests.
  fProxy.cs` file for the template (`SVDWorkspaceTests`, `SvdThinValuesWorkspaceTests`,
  `LQWorkspaceTests`).
- A test in a multiplying template file (`.fProxy.cs`/`.iProxy.cs`) that needs a DIFFERENT expected
  value per generated type must use the SAME per-type mechanism the code under test uses (e.g.
  `//+choose[...]`) for its own expected value too — a hardcoded literal is silently wrong for every
  generated variant except the one it happened to be written for.
