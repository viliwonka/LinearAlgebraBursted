# Spec: unify triangle-trust convention on LOWER (coherence-audit §P.2)

Status: APPROVED (owner ruling 2026-07-11: real unification required, side chosen by
row-major-stride performance; doc-cross-ref-only fix rejected).

## Decision

The whole library trusts/authors the **LOWER triangle + diagonal** for symmetric matrices,
dense and sparse alike. Dense (CHO, CHOP — LAPACK 'L') already does; sparse symmetric BSR
storage (`ToBSRSymmetric`) flips from upper-block canonical to **lower-block canonical**.

## Why lower (row-major analysis)

- Dense row-major Cholesky and its triangular solves are lower-optimal: the inner dot
  products run over two contiguous ROWS (`L[i][0..j)·L[j][0..j)`). Upper in row-major would
  stride column-wise. Dense cannot flip without a real perf loss → dense side is fixed.
- Sparse symmetric spMV/spMM is side-neutral: per stored off-diagonal block it does one
  gather (`y_i += K·x_j`) + one transpose-scatter (`y_j += Kᵀ·x_i`) regardless of which
  triangle is stored. The kernels (`bsrMatVecSym*`, `bsrMatMatSym*`) contain no side
  assumption — only `if (bi != bj)`. Flipping the sparse side is perf-free.
- IC(0) gets a real win: its factor pattern IS A's lower block pattern. With lower-canonical
  symmetric storage, a symmetric-stored SPD matrix feeds `fProxyIC0` with **zero mirror**
  (today it pays a full 2×Nnzb mirror-to-full copy, then reads only the lower half of it).
- Row-prefix property: in lower-canonical storage each block-row's strictly-lower entries
  are a prefix and the diagonal is last (ascending ColInd) — the layout `sweepLower` wants,
  should symmetric-storage sweeps ever be added.
- One authoring rule for users everywhere: "fill lower triangle + diagonal".

## Changes (templates only — `Assets/LinearAlgebra/CodeGen/TemplateSource*`; regen after)

### Production

1. `Sparse/fProxyBSRBuilder.cs` — `ToBSRSymmetric`:
   - Guard flips: reject `blockCol > blockRow` (upper-triangle triplet), accept
     `blockCol <= blockRow`. Error message updated accordingly (static literal,
     ArgumentException, mirror the current message's helpfulness: "add the block at its
     transpose position or use ToBSR()").
   - Diagonal-symmetry check: logic unchanged; message text s/upper/lower/.
   - XML doc: symmetric build now stores lower triangle + diagonal; strictly-UPPER blocks
     are implicit transposes.
2. `Sparse/fProxyBSR.cs` — `Symmetric` doc: lower-block canonical storage.
3. `Sparse/UnsafeOP.Sparse.fProxy.cs` — comments only: `bsrMatVecSym` (general + B1/B2/B3/
   B4/B6), `bsrMatMatSym` (general + B*) — the stored triangle is now lower (`bi >= bj`),
   the implicit mirrored half is the strictly-upper one. NO code changes (verify while
   editing: the kernels must contain no ordering/side assumption; they don't today).
4. `Sparse/Arena.Sparse.fProxy.cs` — `fProxyBSRMirrorToFull`: code is side-agnostic
   (adds each off-diag block + its transpose), keep as-is; doc comments s/upper/lower/.
5. `Sparse/fProxyIC0.cs` — THE functional change: delete the mirror
   (`var A = a.Symmetric ? arena.fProxyBSRMirrorToFull(in a) : a;` → use `a` directly).
   Rationale: symmetric-lower storage pattern == the lower pattern IC0 extracts; the
   existing lower-pattern extraction/`CopyLowerFromA` filter (`bj <= bi`) passes every
   stored block of a symmetric-lower matrix, so the same code path serves both inputs.
   Verify: "every diagonal block stored" validation still fires for symmetric input;
   diagonal blocks arrive full+symmetric per builder contract (CholBlockLower zeroes the
   upper half as today). Update the struct doc: symmetric-storage input is now ZERO-COPY
   (no mirror); full-storage input unchanged.
6. `Sparse/fProxyILU0.cs` — keep the mirror (ILU0 needs both triangles); comment
   s/upper/lower/ where it describes symmetric storage.
7. `Sparse/fProxySSOR.cs` — keep the mirror (sweeps need both triangles row-ordered);
   docs s/upper-only/lower-only/ where describing why the mirror exists.
8. `Sparse/SparseOP.fProxy.cs` — error/doc text mentioning "upper-block-only storage"
   (columnNormsSquared throw message, sweepLower/sweepUpper throw messages, spMVT comment):
   s/upper/lower/. No logic changes.
9. `Sparse/SparseOP.Transpose.fProxy.cs` — symmetric transpose path: verify it stays a
   plain pattern copy (transpose of a symmetric matrix is itself; lower-canonical stays
   lower-canonical). Update comments that say upper.
10. `Sparse/Debug.Sparse.fProxy.cs` (~line 72) — the implicit-half print lookup:
    `if (m.Symmetric && br > bc)` flips to `br < bc` (implicit half is now strictly-upper).
11. `Sparse/fProxySparseLP.fProxy.cs` (~line 305) — inspect the `A.Symmetric` branch;
    update side references; verify no side-dependent logic.
12. `Sparse/Export.Sparse.fProxy.cs` — check for symmetric handling; same treatment as 10
    if it reconstructs the implicit half.
13. `Sparse/fProxyBlockJacobi` (wherever diag extraction lives) — verify diag-block lookup
    scans for `ColInd == bi` (side-agnostic) rather than assuming diag-first. Fix if it
    assumes a position: in lower-canonical rows the diagonal is the LAST entry, not first.

### Tests (`TemplateSourceTests/fProxy`)

Flip every symmetric-authoring site from upper to lower triplets (author at (bigger,
smaller) block coords now): SparseSymmetricTests (also flip the guard tests:
`..._LowerTriangleTriplet_Throws` becomes upper-triplet-throws — rename accordingly),
SparseUnrollTests, SparseTransposeTests, SparseSpMMTests, SparseIC0Tests, SSORTests,
KrylovRound2Tests, SparseStructuralTests, JacobiPrecondTests (1x1 diag — likely unchanged),
DebugPrintTests, plus any other `ToBSRSymmetric` caller a grep finds. Comparison tolerances:
sym-vs-full spMV accumulation order changes (the stored off-diag block for logical (i,j),
i<j is now Kᵀ at (j,i)); existing tolerances should absorb it — do NOT weaken a tolerance
to make a test pass; if one fails investigate first.

NEW test (SparseIC0Tests): symmetric-storage input produces the same factor/preconditioner
behavior as full-storage input on the same matrix (exists as SymmetricStorageMatchesFull —
keep green; it now exercises the zero-mirror path) AND assert no mirror is needed — i.e.
the path works on a matrix whose symmetric storage is the only copy.

### Benchmarks (`TemplateSourceBenchmarks`)

SparseSolverBenchmark (sym builder loop ~line 402-460 authoring comment + triplet side),
LOBPCGBenchmark if it builds symmetric storage. No new sections.

### Docs

- `docs/audit/coherence-audit.md` §P.2: mark RESOLVED (lower-canonical everywhere, date).
- `docs/features/*.md`: grep for sparse symmetric storage side mentions; minimal factual
  edits only (small diffs — no prose rewrites).
- `Sparse/DEVLOG.md`: entry — why lower won (row-major dense fixed side, spMV neutral,
  IC0 zero-mirror), date 2026-07-12.

## Acceptance

- `Tools/regen.ps1` clean; generated float/double/int outputs updated.
- Filtered sparse test classes green (SparseSymmetricTests, SparseIC0Tests, SSORTests,
  SparseUnrollTests, SparseSpMMTests, SparseTransposeTests, SparseStructuralTests,
  KrylovRound2Tests, JacobiPrecondTests, DebugPrintTests) — run via Tools/run-tests.ps1
  with a filter; do NOT run the full suite (main loop does that).
- No public API renames beyond error-message text; `ToBSRSymmetric` name unchanged.
- No commits from the agent.

## Binding rules

- Templates are the source of truth — never hand-edit `Assets/LinearAlgebra/Source`,
  `SourceTests/Generated`, `Benchmarks/Generated`.
- Comment policy: contracts only in code; history/rationale goes to Sparse/DEVLOG.md.
- fProxy discipline: no float literals in templates, `/*+choose[..|..]*/` for per-dtype
  literals.
- Assert.IsTrue with `==` (BC1330), CompileSynchronously=true stays.
