# Spec: alias the SIMD vector proxies to real Unity types, delete the shim layer

Status: pilot proven (QueryOP, commit 49a445f — suite 6317/6317, benchmark no regression). This spec
rolls the pattern out and, at the end, deletes the `fProxyM`/`floatM`/`doubleM` shim + the unused vector
stubs.

## 1. The idea (why this works)

Templates use `fProxy4` as a placeholder that codegen rewrites to `float4` (float file) / `double4`
(double file). Historically `fProxy4` is a hand-written **stub struct** (`proxyStructs.math.cs`) with just
enough surface to compile the template (`+ - * /`, indexer, and `fProxyM.abs/max` for the `math.*` calls
it can't make on the stub type).

But `float4`/`double4` are **real Unity.Mathematics types** that already have every operator AND take
`math.abs/max/min/select/...` natively. So instead of the stub, a file can just alias the token:

```csharp
//+deleteThis
using fProxy4 = Unity.Mathematics.float4;   // codegen deletes this line; body's `fProxy4` token -> float4/double4
//-deleteThis
```

Now in the template `fProxy4` IS `float4`, so `v < best`, `math.select(fProxy4,...)`, `*(fProxy4*)ptr`,
`math.abs(v)` all resolve **natively** — no `fProxyM` indirection. Codegen still swaps the token to
`double4` in the double file (alias line is `deleteThis`'d), where the same `math.*` are native. Generated
`.cs` is byte-identical at runtime (it always used `float4`/`double4`; the stub only ever compiled the
template). Proven end-to-end on QueryOP.

## 2. Per-file recipe

1. Replace `using LinearAlgebra.mathProxies;` (usually already in a `//+deleteThis` block) with
   `//+deleteThis`\n`using fProxy4 = Unity.Mathematics.float4;`\n`//-deleteThis`. (A file can have the
   alias OR the `mathProxies` import, NEVER both — they both define `fProxy4`.)
2. Replace `fProxyM.abs(x)` -> `math.abs(x)`, `fProxyM.max(a,b)` -> `math.max(a,b)`. (And any
   `fProxyM.min`/`select` if present, though none ship today outside the reverted pilot.)
3. Leave `int4`/`bool4`/`math.select(int4,...)` untouched (already real types).
4. `Tools/regen.ps1` + full `Tools/run-tests.ps1` (expect 6317/6317) + spot-benchmark the file's kernels
   to confirm **bit-identical numerics and no perf regression**. One commit per file.

## 3. File inventory + phasing (verified 2026-07-17)

`fProxy4` users: `Arena/ArenaConversions.fProxy.cs`, `OP/LP.RevisedSimplex.fProxy.cs`,
`OP/QueryOP.fProxy.cs` (DONE), `OP/UnsafeOP.fProxy.cs`, `OP/WideOP.fProxy.cs`.
`fProxyM.abs/max` users: `OP/UnsafeOP.fProxy.cs`, `OP/WideOP.fProxy.cs` (ONLY these two).
`fProxyW` users among them: `OP/UnsafeOP.fProxy.cs`, `OP/WideOP.fProxy.cs`.

### Phase 0 — DONE
- `QueryOP.fProxy.cs` (49a445f). Reference implementation.

### Phase 1 — clean pure-`fProxy4` files (no fProxyW)
- `OP/LP.RevisedSimplex.fProxy.cs` — uses `fProxy4` only (no fProxyW, no matrix proxies). Cleanest next.
  Apply the recipe, suite + LPBenchmark.

### Phase 1b — `fProxy4` entangled with the matrix proxies
- `Arena/ArenaConversions.fProxy.cs` — uses `fProxy4` AND matrix proxies (`fProxy4x4` etc.). The matrix
  stubs (`fProxy4x4` has `fProxy4 c0;` fields) depend on the `fProxy4` STRUCT, so a bare `fProxy4` alias
  in this file collides. Do this together with Phase 3 (alias the matrix proxies too) or leave it on the
  stub until then.

### Phase 2 — matrix proxies (separate sub-refactor)
- Alias `fProxy4x4`->`float4x4`/`double4x4`, `fProxy2x2`->`float2x2`, etc. (all real Unity types). VERIFY
  the matrix templates only use ops `float4x4` has natively (mul via `math.mul`, operators, indexer). This
  is the one part not yet proven — check before committing. Unblocks deleting the matrix stubs +
  Phase 1b.

### Phase 3 — ⚠️ GATED ON USER GUIDANCE: the fProxyW files
- `OP/UnsafeOP.fProxy.cs` and `OP/WideOP.fProxy.cs` use **`fProxyW`** (the wide 8-lane / 4-double type).
  **DO NOT touch `fProxyW` or auto-convert these files without the owner's sign-off.** `fProxyW` is NOT
  aliasable — Unity has no `float8`, so it genuinely IS a hand-rolled `v256` wrapper (`WideOP.fProxy.cs`);
  its ops stay custom intrinsics. The `fProxy4`-tail portions of UnsafeOP are *technically* aliasable
  independently of the `fProxyW` paths, but given the risk, treat both files as owner-gated.
- Consequence: these two are the ONLY `fProxyM.abs/max` consumers, so the Phase-4 shim deletion is
  BLOCKED until they convert (or until their `fProxyM.abs/max` calls are swapped to `math.abs/max` with
  the `fProxyW` code left strictly untouched).

### Phase 4 — delete the shim layer (after Phases 1–3)
- Once no file calls `fProxyM.*`: delete class `fProxyM` from `proxyStructs.math.cs` and delete
  `OP/SimdMath.cs` (`floatM`/`doubleM`) entirely.
- Delete the now-unused vector stubs (`fProxy2`/`fProxy3`/`fProxy4`) from `proxyStructs.math.cs` ONLY once
  the matrix stubs (Phase 2) no longer reference them.

## 4. Gotchas / do-not-break

- **Alias vs import collision**: never keep both `using fProxy4 = float4` and `using mathProxies` in one
  file.
- **`fProxyW` is out of scope** — never alias it; never edit its code in this refactor without owner sign-off.
- **Matrix-stub dependency**: the `fProxy4` struct can't be deleted until the matrix proxies are aliased.
- **Per-file suite gate + benchmark**: every file must stay 6317/6317 and bit-identical; the generated
  `.cs` changes only `fProxyM.abs`->`math.abs` (inlines identically) — any numeric or perf drift is a bug.
- **`deleteThis` the alias line** so it doesn't survive into generated code (where the token is already
  `float4`/`double4`).

## 5. Value

Deletes the entire `fProxyM`/`SimdMath.cs` shim + the vector stubs; templates read like normal
Unity.Mathematics (`math.abs(v)`, not `fProxyM.abs(v)`); future SIMD kernels need no indirection; zero
runtime change. The only survivor is `fProxyW` (the genuine "float8"), which has to stay real code.
