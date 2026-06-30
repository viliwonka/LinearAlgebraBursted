# Spec: Predicate-Filtered Queries -- QueryOP extension

Status: **SPEC** (2026-06-28). Promotes the "Tier C" sketch in docs/spec-query.md (Group 4 footer
and the masked-nearest gap flagged in the Call-site validation section) to a coder-ready
implementation spec. All operators live in fProxyQuery_OP (partial class) and follow every
cross-cutting policy from docs/spec-query.md: camelCase, row+col symmetry, Indices+count
convention, zero-alloc, no managed allocs, Burst struct-functor pattern.

---

## 1. Motivation and context

docs/spec-query.md explicitly flags two unresolved gaps:

- **Tier C, Group 4 footer:** predicate functor sketch (IfProxyPredicate, IfProxyRowScore)
  described as "build when a real use appears."
- **Call-site validation gap:** "Masked / predicate-filtered nearest (closest visible enemy) has no
  direct API -- it is the intersection of Group 3 (search) and Tier C (predicate)."

The use case is now concrete: utility-AI needs to filter candidates by a struct-functor predicate
(line-of-sight, faction, health threshold) before running nearest/top-k search. This extension
fills both gaps in one session.

---

## 2. Interface design decisions

### 2a. Pass matrix+index, not an extracted fProxyN

The docs/spec-query.md sketch used Test(in fProxyN row) -- passing an extracted row-vector.
This is rejected because:

- fProxyN is a value type with its own UnsafeList<fProxy> storage. Constructing one from a matrix
  row requires a data copy. Strided columns require a copy too.
- Either extraction path needs a scratch allocation per outer call, breaking zero-alloc without
  adding an Arena parameter to every signature.
- The existing internal metric kernels RowScore(in A, r, in q, m) and ColScore(in A, c, in q, m)
  in QueryOP.fProxy.cs loop directly over A[row, c] / A[r, col] without any extraction.

**Decision: predicate and score interfaces take (in fProxyMxN A, int index), not in fProxyN.**
Zero-alloc, Burst-clean, exactly mirrors RowScore/ColScore. A predicate checking column 2 (health)
above 50 writes A[row, 2] > 50f -- equally readable as any alternative.

### 2b. Two symmetric interface pairs, not one shared interface

Because row predicates access A[row, c] and column predicates access A[r, col], a single shared
interface would need an isRow flag or an overcomplicated contract. Separate named interfaces
(IfProxyRowPredicate, IfProxyColPredicate) bake the axis into the type, matching every existing
row/col op pair in the library. Five interfaces total (see Section 3).

---

## 3. New interfaces

**File (new):**
Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/PredicateQuery.fProxy.cs

Template-expanded (fProxy becomes float / double in generated output). The IfiProxyPredicate
scalar is a non-templated addition at the bottom of the same file.

Interface definitions (exact text the coder must write):

```csharp
namespace LinearAlgebra
{
    // Row predicate: implementation reads A[row, 0..N_Cols-1]. Zero-alloc.
    // where P : struct, IfProxyRowPredicate  -> Burst monomorphizes, zero overhead.
    public interface IfProxyRowPredicate {
        bool Test(in fProxyMxN A, int row);
    }

    // Column predicate: symmetric twin. Reads A[0..M_Rows-1, col] with stride.
    public interface IfProxyColPredicate {
        bool Test(in fProxyMxN A, int col);
    }

    // Scalar / elementwise predicate for flat IUnsafefProxyArray data.
    public interface IfProxyPredicate {
        bool Test(fProxy x);
    }

    // Row-score functor: returns a scalar score for row r of A.
    public interface IfProxyRowScore {
        fProxy Score(in fProxyMxN A, int row);
    }

    // Column-score functor: symmetric twin.
    public interface IfProxyColScore {
        fProxy Score(in fProxyMxN A, int col);
    }

    // Scalar predicate for integer flat data. Not template-expanded.
    public interface IfiProxyPredicate {
        bool Test(iProxy x);
    }
}
```

---

## 4. Implementation files

### 4a. Main fProxy operator file

**File (new):**
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.Predicate.fProxy.cs

public static partial class fProxyQuery_OP in namespace LinearAlgebra. Same #define and using
directives as QueryOP.fProxy.cs.

#### Group A: Flat / scalar predicate ops

Constraints on all five: where T : unmanaged, IUnsafefProxyArray
and where P : struct, IfProxyPredicate.

| Signature | Semantics |
|-----------|-----------|
| int findFirst<T,P>(in T x, ref P pred) | First flat index i where pred.Test(x.Data[i]) is true; returns -1 if none. Short-circuits. |
| int count<T,P>(in T x, ref P pred) | Count of elements where pred is true. Full scan. |
| bool any<T,P>(in T x, ref P pred) | True if at least one element satisfies pred. Short-circuits on first true. |
| bool all<T,P>(in T x, ref P pred) | True if every element satisfies pred. Short-circuits on first false. |
| int findAll<T,P>(in T x, ref P pred, ref Indices idx) | Fills idx[0..count) with flat indices where pred.Test is true. Returns count. |

Edge cases for all five: empty x.Data.Length == 0 -> findFirst returns -1, count returns 0,
any returns false, all returns true (vacuous), findAll returns 0 with no writes. No throw on empty.

Guard for findAll: throw ArgumentException("QueryOP.findAll: idx.N must be >= x.Data.Length")
when idx.N < x.Data.Length.

#### Group B: Row / column filter

| Signature | Constraint | Semantics |
|-----------|------------|-----------|
| int countRows<P>(in fProxyMxN A, ref P pred) | where P : struct, IfProxyRowPredicate | Count rows r where pred.Test(in A, r). |
| int whichRows<P>(in fProxyMxN A, ref P pred, ref Indices idx) | same | Fills idx[0..count). Returns count. |
| int countColumns<P>(in fProxyMxN A, ref P pred) | where P : struct, IfProxyColPredicate | Count columns c where pred.Test(in A, c). |
| int whichColumns<P>(in fProxyMxN A, ref P pred, ref Indices idx) | same | Fills idx[0..count). Returns count. |

Guards:
- whichRows: throw ArgumentException("QueryOP.whichRows: idx.N must be >= A.M_Rows") when
  idx.N < A.M_Rows.
- whichColumns: throw ArgumentException("QueryOP.whichColumns: idx.N must be >= A.N_Cols") when
  idx.N < A.N_Cols.
- Empty matrix (0 rows or 0 cols): return 0, no throw.

#### Group C: Masked nearest / k-nearest

These ops run a standard nearest/k-nearest scan but skip candidates where pred returns false.
They call the existing internal helpers RowScore, ColScore, IsBetterForNearest, and
WorstScoreForNearest directly. No metric logic is duplicated.

**Empty-result contract (when zero rows/columns pass pred):**
- nearestRowWhere / nearestColumnWhere: set index = -1 and score = WorstScoreForNearest(m)
  (fProxy.MaxValue for distance metrics, fProxy.MinValue for similarity metrics).
  Callers must check index == -1 before use.
- kNearestRowsWhere / kNearestColumnsWhere: return 0.

| Signature | Constraint | Semantics |
|-----------|------------|-----------|
| void nearestRowWhere<P>(in fProxyMxN A, in fProxyN q, Metric m, ref P pred, out int index, out fProxy score) | where P : struct, IfProxyRowPredicate | Nearest row among rows passing pred. |
| int kNearestRowsWhere<P>(in fProxyMxN A, in fProxyN q, int k, Metric m, ref P pred, ref Indices idx, ref fProxyN scores) | same | k-nearest among passing rows. Returns actual count (<= min(k, M_Rows)). |
| void nearestColumnWhere<P>(in fProxyMxN A, in fProxyN q, Metric m, ref P pred, out int index, out fProxy score) | where P : struct, IfProxyColPredicate | Symmetric column twin. |
| int kNearestColumnsWhere<P>(in fProxyMxN A, in fProxyN q, int k, Metric m, ref P pred, ref Indices idx, ref fProxyN scores) | same | Symmetric column twin. |

Parameter guards for row variants (column twins mirror these, swapping M_Rows/N_Cols):
- nearestRowWhere: throw InvalidOperationException("QueryOP.nearestRowWhere: matrix has no rows")
  if A.M_Rows == 0; throw ArgumentException("QueryOP.nearestRowWhere: q.N must equal A.N_Cols")
  if q.N != A.N_Cols.
- kNearestRowsWhere: return 0 if A.M_Rows == 0 or k <= 0; throw if q.N != A.N_Cols;
  throw if idx.N < k or scores.N < k (prefix "QueryOP.kNearestRowsWhere:").

Algorithm for kNearestRowsWhere: identical bounded-insertion sort as kNearestRows
(QueryOP.fProxy.cs lines 692-712), with one addition: at the top of the outer for-r loop, add
"if (!pred.Test(in A, r)) continue;" before computing RowScore. clampedK is still
math.min(k, A.M_Rows) as an upper bound; the returned count reflects only passing rows found.

#### Group D: Score-based row / column selection

| Signature | Constraint | Semantics |
|-----------|------------|-----------|
| void argMaxRowBy<S>(in fProxyMxN A, ref S scorer, out int index, out fProxy score) | where S : struct, IfProxyRowScore | Row maximizing scorer.Score. Ties: first wins. Throws if M_Rows == 0. |
| void argMinRowBy<S>(in fProxyMxN A, ref S scorer, out int index, out fProxy score) | same | Row minimizing scorer.Score. |
| int topKRowsBy<S>(in fProxyMxN A, ref S scorer, int k, ref Indices idx, ref fProxyN scores) | same | k rows with highest score, best-first. Returns min(k, M_Rows). |
| void argMaxColBy<S>(in fProxyMxN A, ref S scorer, out int index, out fProxy score) | where S : struct, IfProxyColScore | Column maximizing scorer.Score. |
| void argMinColBy<S>(in fProxyMxN A, ref S scorer, out int index, out fProxy score) | same | Column minimizing. |
| int topKColsBy<S>(in fProxyMxN A, ref S scorer, int k, ref Indices idx, ref fProxyN scores) | same | k columns with highest score, best-first. Returns min(k, N_Cols). |

Guards for topKRowsBy: throw ArgumentException("QueryOP.topKRowsBy: idx.N must be >= k") and
matching for scores. argMaxRowBy / argMinRowBy throw
InvalidOperationException("QueryOP.argMaxRowBy: matrix has no rows") when M_Rows == 0.

argMaxRowBy initializes score = fProxy.MinValue before the loop; argMinRowBy uses fProxy.MaxValue.
topKRowsBy uses the same bounded-insertion as kNearestRows with higher-score-wins direction
(no Metric enum; direction is always higher-is-better for argMaxBy/topKBy).

### 4b. iProxy scalar predicate file

**File (new):**
Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.Predicate.iProxy.cs

public static partial class iProxyQuery_OP. Group A only, using IfiProxyPredicate:
findFirst<T,P>, count<T,P>, any<T,P>, all<T,P>, findAll<T,P>.
Constraints: where T : unmanaged, IUnsafeiProxyArray and where P : struct, IfiProxyPredicate.
Logic identical to fProxy Group A with iProxy substituted. Document at file top: "Groups B/C/D
are fProxy-only. For integer matrix row/col filtering use the float or double variant."

### 4c. Arena-allocating wrappers (lower priority -- second commit)

**File (new):**
Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.QueryPredicate.fProxy.cs

Follows the pattern in ArenaExtensions.Query.fProxy.cs. Four wrappers, two-pass count+alloc:
- fProxyWhichRows<P>(this ref Arena arena, in fProxyMxN A, ref P pred) -> Indices
- fProxyWhichColumns<P>(this ref Arena arena, in fProxyMxN A, ref P pred) -> Indices
- fProxyKNearestRowsWhere<P>(..., ref P pred, out fProxyN scores) -> Indices
- fProxyTopKRowsBy<S>(..., out fProxyN scores) -> Indices

Implement Groups A-D and tests first; arena wrappers are a follow-up commit if time allows.

---

## 5. Files to touch (summary)

| Path | Action |
|------|--------|
| Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/PredicateQuery.fProxy.cs | NEW |
| Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.Predicate.fProxy.cs | NEW |
| Assets/LinearAlgebra/CodeGen/TemplateSource/OP/QueryOP.Predicate.iProxy.cs | NEW |
| Assets/LinearAlgebra/CodeGen/TemplateSource/Arena/ArenaExtensions.QueryPredicate.fProxy.cs | NEW (lower priority) |
| Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QueryPredicateTests.fProxy.cs | NEW |

Do NOT edit QueryOP.fProxy.cs, QueryEnums.cs, or any file under Source/Generated/.

---

## 6. Tests

**File (new):**
Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/QueryPredicateTests.fProxy.cs

Outer class fProxyQueryPredicateTests. One TestJob : IJob with a TestType enum; plain [Test]
methods for managed-throw guards. Pattern matches QueryTests.fProxy.cs exactly (the
NativeArray<fProxy> Fail reporter, RunJob(TestType t) helper, etc.).

Define these test functor structs inside the test file:

```csharp
struct GreaterThanScalar : IfProxyPredicate {
    public fProxy t;
    public bool Test(fProxy x) => x > t;
}
struct RowSumAbove : IfProxyRowPredicate {
    public fProxy t;
    public bool Test(in fProxyMxN A, int r) {
        fProxy s = (fProxy)0;
        for (int c = 0; c < A.N_Cols; c++) s += A[r, c];
        return s > t;
    }
}
struct ColSumAbove : IfProxyColPredicate {
    public fProxy t;
    public bool Test(in fProxyMxN A, int c) {
        fProxy s = (fProxy)0;
        for (int r = 0; r < A.M_Rows; r++) s += A[r, c];
        return s > t;
    }
}
struct RowL2Score : IfProxyRowScore {
    public fProxy Score(in fProxyMxN A, int r) {
        fProxy s = (fProxy)0;
        for (int c = 0; c < A.N_Cols; c++) s += A[r,c] * A[r,c];
        return s;
    }
}
struct ColL2Score : IfProxyColScore {
    public fProxy Score(in fProxyMxN A, int c) {
        fProxy s = (fProxy)0;
        for (int r = 0; r < A.M_Rows; r++) s += A[r,c] * A[r,c];
        return s;
    }
}
struct AlwaysTrueRow  : IfProxyRowPredicate { public bool Test(in fProxyMxN A, int r) => true; }
struct AlwaysFalseRow : IfProxyRowPredicate { public bool Test(in fProxyMxN A, int r) => false; }
struct AlwaysTrueCol  : IfProxyColPredicate { public bool Test(in fProxyMxN A, int c) => true; }
struct AlwaysFalseCol : IfProxyColPredicate { public bool Test(in fProxyMxN A, int c) => false; }
```

### T1 -- Flat scalar predicate ops (Group A)

- findFirst on a 6-element vector returns the correct first matching index.
- findFirst returns -1 when no element matches.
- count result matches a manual loop count on the same vector.
- any is true when at least one element passes; false on an all-zero vector.
- all is false when one element fails; true on a vector where every element passes.
- findAll fills indices in ascending scan order; returned count equals the filled prefix length.
- Empty vector (length 0): findFirst -> -1, count -> 0, any -> false, all -> true,
  findAll -> 0 with no idx writes.

### T2 -- Row/col filter (Group B)

- whichRows on a 4x3 matrix with known row sums returns the correct index subset and count.
- countRows on the same matrix equals the count returned by whichRows.
- AlwaysTrueRow: countRows == A.M_Rows, whichRows returns all row indices in order.
- AlwaysFalseRow: countRows == 0, no indices written to idx.
- **Row/col symmetry:** whichColumns(A, colPred) returns the same indices as
  whichRows(Atranspose, rowPred) where both predicates test the equivalent condition.
  (Column j of A is row j of Atranspose.)

### T3 -- Masked nearest / k-nearest (Group C -- highest-value tests)

- Build a 5x2 matrix with five known 2D points. A predicate excludes rows 1 and 3 (e.g.,
  row sum below threshold). Verify nearestRowWhere returns the same row as a brute-force loop
  that skips rows 1 and 3.
- **All-pass predicate (AlwaysTrueRow):** nearestRowWhere returns the same index and score as
  nearestRow on identical inputs. Score equality is exact (same code path, same arithmetic).
- **Empty-result (AlwaysFalseRow):** index == -1 and score == fProxy.MaxValue for SqEuclidean;
  score == fProxy.MinValue for Cosine.
- **k-nearest masked:** kNearestRowsWhere with k=3, 2 rows passing -> returned count == 2;
  both indices and scores match the brute-force masked scan.
- **All-pass k-nearest:** kNearestRowsWhere with AlwaysTrueRow returns the same results as
  kNearestRows on identical inputs (exact float equality -- same insertion sort, same order).
- **Column symmetry:** nearestColumnWhere(A, q, SqEuclidean, colPred) returns the same index as
  nearestRowWhere(Atranspose, q, SqEuclidean, rowPred) for equivalent predicates and a query
  vector of matching dimension.
- Metrics: SqEuclidean (distance, nearest=MIN) and Cosine (similarity, nearest=MAX).

### T4 -- Score-based selection (Group D)

- argMaxRowBy with RowL2Score returns the row with the largest L2-squared norm. Cross-check:
  result must equal argMaxRowNorm(A, Norm.L2) on the same matrix (argmax is monotone under sqrt).
- argMinRowBy with RowL2Score returns the row with the smallest L2-squared norm.
- topKRowsBy with k=2 returns the two rows with the largest squared norm, sorted best-first.
- Empty matrix (0 rows): argMaxRowBy throws InvalidOperationException.
- **Col symmetry:** argMaxColBy(A, ColL2Score) result must equal argMaxColNorm(A, Norm.L2).

### T5 -- Guard / argument checks (plain [Test], main thread)

- findAll with idx.N < x.Data.Length throws ArgumentException with message starting
  "QueryOP.findAll:".
- whichRows with idx.N < A.M_Rows throws ArgumentException starting "QueryOP.whichRows:".
- nearestRowWhere with q.N != A.N_Cols throws ArgumentException starting
  "QueryOP.nearestRowWhere:".
- nearestRowWhere on a 0-row matrix throws InvalidOperationException.
- kNearestRowsWhere with k <= 0 returns 0 without throwing.
- kNearestRowsWhere with idx.N < k throws ArgumentException starting
  "QueryOP.kNearestRowsWhere:".

---

## 7. Acceptance criteria

1. All five new template files in TemplateSource/ are present; codegen regenerates without errors
   or CS* warnings.
2. All T1-T5 test cases exist in QueryPredicateTests.fProxy.cs and pass headlessly via the
   existing Tools PowerShell runner.
3. kNearestRowsWhere(A, q, k, m, AlwaysTrueRow, idx, scores) returns results byte-for-byte
   identical to kNearestRows(A, q, k, m, idx, scores) on the same inputs.
4. nearestRowWhere(A, q, m, AlwaysFalseRow, out int i, out fProxy s) yields i == -1 for any
   valid non-empty matrix and any Metric value.
5. No new, List<>, int[], delegate, lambda, or LINQ anywhere in Groups A-D implementations.
6. The [BurstCompile] attribute on TestJob compiles without managed-type errors.

---

## 8. Out of scope

- No spatial acceleration (k-d tree, grid). Brute-force O(M * d) is the design.
- No farthestRowWhere / farthestColumnWhere.
- Do not modify QueryOP.fProxy.cs, QueryEnums.cs, or any file under Source/Generated/.
- No pairwiseDistancesWhere or other pairwise masked ops.
- No extraction of rows or columns into fProxyN inside any implementation.
- Groups B, C, D are NOT generated for iProxy.
- Arena wrappers (Section 4c) may be deferred to a second commit.

---

## 9. Open design questions for owner

**Q1 -- iProxy Group A unification.** findFirst/count/any/all/findAll in iProxyQuery_OP use
IfiProxyPredicate. If the owner wants a single generic call-site that works on both float and
integer flat data, a shared base predicate interface would be needed, touching the interface
hierarchy more broadly. Current spec keeps them separate, consistent with the existing fProxy/iProxy
split everywhere else. Confirm or redirect.

**Q2 -- findAll vs which naming for flat data.** Group B uses whichRows/whichColumns. Should the
flat-data Group A variant be renamed which<T,P>(in T x, ref P pred, ref Indices idx) for naming
consistency, or does findAll (more search-flavoured, matching the Group 4 vocabulary in
spec-query.md) read better at call sites? Rename-only decision; no implementation impact.
