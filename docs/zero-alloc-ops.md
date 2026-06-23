# Preallocated-destination (zero-alloc) op overloads — spec

Goal: every allocating arithmetic op gains a `ref`-destination overload so hot loops
can run without per-call arena allocations. Source of truth for the loop doing this work.

## Decisions (locked)

- **Scope (phase 1): arithmetic ops only.** The `dot` family, `outerDot`, `trans`, `select`.
  Decompositions/solvers already take preallocated `ref` outputs; eliminating *their*
  internal temporaries (workspace structs) is a separate later phase.
- **Component ops are already done** — `fProxyOP.addInpl/mulInpl/subInpl/compMulInpl/...`
  are generic in-place forms over `IUnsafefProxyArray`, with operator sugar (`+`,`*`,…)
  for the allocating path. Do NOT touch them.
- **Naming:** overload the existing method with a trailing `ref <dest>` parameter
  (e.g. `dot(in A, in x, ref y)`). No new suffix. For `dot(a,b,transposeA)` the dest
  goes before the optional flag: `dot(in a, in b, ref C, bool transposeA = false)`.
- **Primitive + wrapper (hard rule):** the `ref`-dest form is the implementation; the
  existing allocating form becomes a thin wrapper that `tempMat`/`tempVec`-allocates and
  calls the `ref` form. No duplicated math.
- **Operators stay allocating** — C# operator signatures are fixed; they can't take a dest.
- **Dimension validation:** the `ref` form throws `ArgumentException` if the destination
  is mis-sized (matching existing guard style / `Assume`).
- **Templates:** write overloads in `*.fProxy.cs` / `*.iProxy.cs` so codegen emits
  float/double/int/short/long automatically.
- **Tests:** for each new `ref` op, a test asserting the `ref` result equals the
  allocating result on random input (Burst IJob pattern, per-precision `Tol`).
- **Green per chunk:** one OP file at a time → `regen.ps1` + `run-tests.ps1`, never red.

## Aliasing guard

Rule: **inputs may alias each other freely (read-only); only the destination aliasing an
input is forbidden — and only for ops where an input element feeds multiple outputs or a
moved output (contracting / permuting ops).** Elementwise ops keep allowing dest=input
(that's the in-place use). Guard compares `dest.Data.Ptr` against each input pointer only;
never input-vs-input.

| Op | Dest type | Guard |
|----|-----------|-------|
| `dot(A, x, ref y)` mat·vec | vector | `y` vs `x` (matrix input is type-disjoint) |
| `dot(y, A, ref x)` vec·mat | vector | dest vs `y` |
| `dot(A, B, ref C)` mat·mat | matrix | `C` vs `A`, `C` vs `B` |
| `trans(A, ref At)` | matrix | `At` vs `A` (permutation → unsafe even though read-once) |
| `outerDot(a, b, ref C)` | matrix | none — inputs are vectors, can't share a matrix buffer |
| `select(a, b, c, ref d)` | vec/mat | none — elementwise, dest may alias `a`/`b` |
| scalar `dot(a,b)`, norms, `determinant`, `trace` | — | no overload (already alloc-free) |

Guard form (inside the `ref` op, after dim checks):
```csharp
unsafe {
    if (C.Data.Ptr == a.Data.Ptr || C.Data.Ptr == b.Data.Ptr)
        throw new ArgumentException("dot: destination must not alias an input");
}
```
Always-on (cost is two pointer compares vs an O(n^k) kernel). Exact `==` suffices because
all allocations are whole-matrix/vector (no slices/views in scope).

## Inventory (check off as done)

- [x] `OP/OP.Dot.fProxy.cs` — `dot` mat·vec, vec·mat, mat·mat(+transA); `outerDot`; `trans`
- [x] `OP/OP.Dot.iProxy.cs` — same set for int/short/long
- [x] `OP/SelectOP.fProxy.cs` — `select` vec, mat, scalar-cond variants (no guard)
- [x] Tests: `DotRefTests.fProxy.cs` — equivalence + transpose oracle + DirtyDest (reused-dest)
- [~] Tests: `DotRefTests.iProxy.cs`, `SelectRefTests.fProxy.cs` (written, validating)
- [~] Tests: `DotRefGuardTests.fProxy.cs` — managed Assert.Throws for dim + alias guards (validating)
- [x] bug-hunter pass on the guards + dim checks (opus)
- [ ] README/docs note

Done = every box checked, suite green, allocating forms delegate to ref forms.

## Bugs found & fixed (surfaced by this work)

1. **CRITICAL — accumulating ref forms didn't zero the destination.** `matVecDot`/`vecMatDot`/
   `matMatDot(TransA)` use `+=` and require a zeroed output; the allocating forms got it free
   (temp alloc clears), the ref forms didn't → a reused destination accumulated garbage. Fixed
   with `UnsafeUtility.MemClear` in the 3 accumulating ref forms (fProxy+iProxy). Regression test
   = `DirtyDest`. Caught by bug-hunter, not by the equality test (fresh buffers are zeroed).
2. **`dot(a,b,transposeA:true)` checked the wrong axis** — `a.N_Cols==b.N_Cols` instead of
   `a.M_Rows==b.M_Rows` (Aᵀ·B contracts over rows). Rejected valid non-Gram inputs. Fixed.
3. **`Assume.SameDim(matrix,matrix)` used `&&`** — only threw if BOTH dims differed; a one-axis
   mismatch slipped through into OOB reads. Fixed to `||` (5 sites: Assume.cs/fProxy/iProxy).
4. **`Assume.IndexInsideBounds` was wrong** — `math.any(0 < index) && math.any(index >= dim)`
   missed negative indices and most overflows. Fixed to `math.any(index < 0) || math.any(index >= dim)`.
   It guards every `matrix[r,c]` under ENABLE_UNITY_COLLECTIONS_CHECKS.
5. Tooling: `run-tests.ps1` failure reporter crashed on XmlElement messages; `-Filter` is a regex
   (glob `*` now auto-converted to `.*`).
```
