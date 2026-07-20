# Spec: standardized declarative test battery for the Krylov solver family

Status: design spec only (no test/production code in this document). Target: a `coder`/
`test-writer` agent implements this in one or more follow-up sessions per the migration
sequence in SS8.

## 0. Ground-truth check (read this before implementing)

The task brief that motivated this spec assumed a few things about solver inventory that do
**not** match the current codebase. Verified directly against
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov*.fProxy.cs` and
`docs/dev/spec-square-noncg-survey.md`:

- **TFQMR and GCRO-DR are NOT implemented.** They are roadmap-only (`docs/dev/spec-square-
  noncg-survey.md` SS2b/SS2c, priority-ranked future work). There is no `Krylov.tfqmr` /
  `Krylov.gcrodr` to wire.
- **CRAIG / CRAIGMR were implemented and then explicitly REMOVED** (`OP/DEVLOG.md`: "2026-07-
  18 | Removed CGNE / Craig's method (all overloads: generic + dense + BSR) + its tests"). Do
  not resurrect them as part of this task.
- **The block-solver family currently has 6 members, not 9**: `bcg`, `bcgrq`, `bfbcg`,
  `bbiCGStab`, `bgmres`, `bminres` (`Krylov.Block.{BCGrQ,BFBCG,BiCGStab,CG,GMRES,MINRES}.
  fProxy.cs`). There is no 7th/8th/9th block solver today.
- **`bminres` IS implemented and already has a test file** (`BlockMinresTests.fProxy.cs`,
  345 lines) -- `docs/dev/spec-bminres-fix.md` documents its historical bug (identical RHS rows
  silently diverging, fixed by a `Beta^T` correction in `BuildOmega`). This is exactly the bug
  the "identical RHS columns" invariant in SS5.3 is designed to catch as a standing regression
  guard, not a bug still open today.
- **`Krylov.MINRESQLP` (`minresQLP<TOp,TPre>` / `minresQLP<TOp>`) IS implemented but has ZERO
  existing tests** -- no `KrylovMinresQLPTests.fProxy.cs` or equivalent exists anywhere under
  `TemplateSourceTests`. This is genuine new coverage, not a migration.

Net effect on scope: SS8's migration plan wires the **currently-implemented** solver inventory
(9 square, 6 block, 2 least-squares -- see SS8.1's table) into the battery, backfills
MINRES-QLP from zero, and explicitly leaves TFQMR/GCRO-DR/CRAIG/CRAIGMR as "wire this in when
it ships" future work, not part of this task.

## 1. Problem

Each Krylov solver has its own ~230-340-line bespoke test file
(`BlockCGTests.fProxy.cs`, `BlockGmresTests.fProxy.cs`, `BlockBiCGStabTests.fProxy.cs`,
`FGMRESTests.fProxy.cs`, `IDRTests.fProxy.cs`, `GMRESTests.fProxy.cs`,
`KrylovPMinresTests.fProxy.cs`, etc. under `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/
fProxy/`) re-implementing the same handful of scenarios -- converges, matches a known solution,
identity-fold, preconditioned convergence, determinism -- with duplicated per-file helpers
(`RelResidualDense`/`RelResidualBSR`/`DenseNonsym`/`ConvDiff1D` in `IDRTests.fProxy.cs`;
`BuildDenseSPD`/`Row`/`DenseToBSR1x1` in `BlockCGTests.fProxy.cs`; etc). Two existing files
already demonstrate the target shape for this kind of consolidation and are the direct
precedent this spec follows:

- `SolverBatteryTests.fProxy.cs` -- one `[BurstCompile] IJob` with a `TestType` enum, one
  `[TestCaseSource]` NUnit entry point, `Consts.fProxySqrtEps`-scaled per-precision tolerances
  via a local `IsDouble()` helper, and a `NativeArray<fProxy> Fail` diagnostic payload that
  survives past the Burst boundary so a failing case reports `(got, expected, diff)` instead of
  just "false". Drives decompositions + `Krylov.cg` only.
- `PreconditionerBatteryTests.fProxy.cs` -- the same shape, one level narrower: preconditioner x
  BSR-matrix cross-coverage, symmetric-M routed through `cg`, nonsymmetric-M through `biCGStab`.

This spec generalizes both into a solver x matrix-regime cross-coverage battery covering the
whole Krylov family (square, block, least-squares), while keeping genuinely solver-specific
behavior (restart/recycling, seeded-shadow determinism, min-length singular solutions) in small
residual files.

## 2. Matrix-property profile

### 2.1 The flag set

New type-agnostic (no `fProxy` token) file:
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/KrylovBatteryProfile.cs` -- a **singular**
file (codegen copies it as-is to both `SourceTests/Generated/float` and `.../double`, same
mechanism that already applies to `RandomSharedTests.cs` at the same directory level: no
`fProxy`/`iProxy` token anywhere in the file, so `TemplateConverter` treats it as singular
without needing the explicit `//singularFile//` marker).

```csharp
namespace LinearAlgebra
{
    /// <summary>
    /// Tags a Krylov battery gallery matrix (what it structurally IS) and, on the solver side,
    /// what a solver family REQUIRES/FORBIDS of a matrix it is willing to run against. Two
    /// disjoint sub-groups:
    ///   KIND   (mutually exclusive per matrix): SPD, SymmetricIndefinite, Nonsymmetric.
    ///   SHAPE  (mutually exclusive per matrix): Square, Rectangular (+ Overdetermined /
    ///          Underdetermined as a Rectangular refinement).
    ///   MODIFIER (orthogonal, any combination): FullRank, RankDeficient, WellConditioned,
    ///          IllConditioned, Sparse (BSR-native vs dense literature-gallery).
    /// Every gallery matrix carries exactly one KIND flag, exactly one SHAPE flag (plus
    /// Overdetermined/Underdetermined when Rectangular), and any applicable MODIFIER flags.
    /// </summary>
    [System.Flags]
    public enum MatrixProfile : uint
    {
        None                 = 0,

        // KIND (exactly one per square matrix; rectangular matrices don't carry a KIND flag)
        SPD                  = 1 << 0,
        SymmetricIndefinite  = 1 << 1,
        Nonsymmetric         = 1 << 2,

        // SHAPE (exactly one of Square/Rectangular; Over/Under only set when Rectangular)
        Square               = 1 << 3,
        Rectangular          = 1 << 4,
        Overdetermined       = 1 << 5,
        Underdetermined      = 1 << 6,

        // MODIFIERS (orthogonal)
        FullRank             = 1 << 7,
        RankDeficient        = 1 << 8,
        WellConditioned      = 1 << 9,
        IllConditioned       = 1 << 10,
        Sparse               = 1 << 11,   // BSR-native gallery entry (unlocks the
                                           // preconditioned-convergence check; dense entries
                                           // never carry this flag)
    }

    /// <summary>Which preconditioner a solver invoker expects for the Sparse-only
    /// preconditioned-convergence check (SS5.2 #5). Mirrors the symmetric/nonsymmetric routing
    /// PreconditionerBatteryTests already uses (BlockJacobi for cg-family, ILU0 for
    /// biCGStab/gmres/idr-family).</summary>
    public enum PreconditionerKind { None, SymmetricBSR, NonsymmetricBSR }
}
```

### 2.2 Match rule (refinement of the brief's "tags intersect")

A plain flags-intersection is **wrong** for the KIND group: `Square` alone would spuriously
match `cg` (which needs `SPD`) against a `Nonsymmetric`-tagged matrix, since both carry `Square`
and intersection would be non-empty. Each solver invoker instead declares a `Requires` set
(every one of these flags MUST be present on the matrix) and a `Forbids` set (none of these
flags may be present):

```csharp
static bool Applicable(MatrixProfile requires, MatrixProfile forbids, MatrixProfile matrixTags)
    => (matrixTags & requires) == requires && (matrixTags & forbids) == MatrixProfile.None;
```

Example: `cg.Requires = SPD`, `cg.Forbids = None` -> matches only `SPD`-tagged matrices.
`idr.Requires = Square`, `idr.Forbids = None` -> matches `SPD | Square`,
`SymmetricIndefinite | Square`, and `Nonsymmetric | Square` matrices alike (IDR is
general-purpose). `biCGStab.Requires = Square`, `biCGStab.Forbids = None` -- same breadth as
`idr` (biCGStab is also usable on any square system, not just nonsymmetric ones; the dense
literature gallery already tags SPD matrices this way for cross-solver comparison in the special
files, SS7).

### 2.3 Gallery tagging registry

Two more enums in the same singular `KrylovBatteryProfile.cs` file (pure enum + pure switch,
zero `fProxy` dependency -- construction of the actual matrices is templated separately, SS2.4):

```csharp
public enum GalleryDenseMatrix
{
    // SPD
    Laplacian1D_8, MinIJ_5, Pei5_2, Hilbert4, Pascal5, Lehmer5,
    // SymmetricIndefinite
    Fiedler5, Clement4, Rosser8,
    // Nonsymmetric (square)
    DenseNonsym20, ConvDiffDense40, Grcar8,
    // Rectangular (Overdetermined) -- least-squares family
    Lauchli3_05, Lauchli3_1e3,
    // Rectangular (Underdetermined) -- least-squares family
    WideRandom10x30,
    // Rank-deficient rectangular -- least-squares family
    RankDeficient20x10_Rank5,
    // Synthetic modifiers (Rand.*InPlace) -- clean, size-independent WellConditioned /
    // IllConditioned knobs the literature gallery doesn't give directly
    RandSPDWellCond20, RandSPDIllCond20,
}

public enum GalleryBSRMatrix
{
    Poisson2D_20x20,          // SPD, Sparse
    Laplacian2D_16x16,        // SPD, Sparse
    RandomSparseSPD_120_2,    // SPD, Sparse
    RandomSparseNonsym_80,    // Nonsymmetric, Sparse
}
```

```csharp
public static class GalleryProfiles
{
    public static MatrixProfile Of(GalleryDenseMatrix m)
    {
        switch (m)
        {
            case GalleryDenseMatrix.Laplacian1D_8: return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
            case GalleryDenseMatrix.MinIJ_5:        return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
            case GalleryDenseMatrix.Pei5_2:          return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
            case GalleryDenseMatrix.Hilbert4:        return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;
            case GalleryDenseMatrix.Pascal5:         return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;
            case GalleryDenseMatrix.Lehmer5:         return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;

            case GalleryDenseMatrix.Fiedler5:  return MatrixProfile.SymmetricIndefinite | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
            case GalleryDenseMatrix.Clement4:  return MatrixProfile.SymmetricIndefinite | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
            case GalleryDenseMatrix.Rosser8:   return MatrixProfile.SymmetricIndefinite | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;

            case GalleryDenseMatrix.DenseNonsym20:   return MatrixProfile.Nonsymmetric | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
            case GalleryDenseMatrix.ConvDiffDense40: return MatrixProfile.Nonsymmetric | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
            case GalleryDenseMatrix.Grcar8:          return MatrixProfile.Nonsymmetric | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;

            case GalleryDenseMatrix.Lauchli3_05:  return MatrixProfile.Rectangular | MatrixProfile.Overdetermined | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
            case GalleryDenseMatrix.Lauchli3_1e3: return MatrixProfile.Rectangular | MatrixProfile.Overdetermined | MatrixProfile.FullRank | MatrixProfile.IllConditioned;

            case GalleryDenseMatrix.WideRandom10x30: return MatrixProfile.Rectangular | MatrixProfile.Underdetermined | MatrixProfile.FullRank | MatrixProfile.WellConditioned;

            case GalleryDenseMatrix.RankDeficient20x10_Rank5: return MatrixProfile.Rectangular | MatrixProfile.Overdetermined | MatrixProfile.RankDeficient | MatrixProfile.WellConditioned;

            case GalleryDenseMatrix.RandSPDWellCond20: return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned;
            case GalleryDenseMatrix.RandSPDIllCond20:  return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.IllConditioned;

            default: return MatrixProfile.None;
        }
    }

    public static MatrixProfile Of(GalleryBSRMatrix m)
    {
        switch (m)
        {
            case GalleryBSRMatrix.Poisson2D_20x20:       return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned | MatrixProfile.Sparse;
            case GalleryBSRMatrix.Laplacian2D_16x16:     return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned | MatrixProfile.Sparse;
            case GalleryBSRMatrix.RandomSparseSPD_120_2: return MatrixProfile.SPD | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned | MatrixProfile.Sparse;
            case GalleryBSRMatrix.RandomSparseNonsym_80: return MatrixProfile.Nonsymmetric | MatrixProfile.Square | MatrixProfile.FullRank | MatrixProfile.WellConditioned | MatrixProfile.Sparse;
            default: return MatrixProfile.None;
        }
    }
}
```

Exact sizes/parameters for the `Rand*` synthetic entries and the two new `WideRandom10x30` /
`RankDeficient20x10_Rank5` generators are the implementer's call (use
`Rand.spdInPlace`/`Rand.conditionedInPlace`/`Rand.withRankInPlace` -- all already exist in
`OP/RandomMatrixOP.fProxy.cs`, namespace `LinearAlgebra.Rand`, no Gallery opt-in needed); pick
values that keep the battery's total runtime in the same ballpark as `SolverBatteryTests`
today.

### 2.4 Building the tagged matrices (templated, per-type)

New file: `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovBattery.Gallery.fProxy.cs`
(templated -- constructs real `fProxyMxN`/`fProxyBSR` values, so it needs the `fProxy` token and
is generated once per numeric type):

```csharp
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    internal static class fProxyKrylovBatteryGallery
    {
        public static fProxyMxN Build(ref Arena arena, GalleryDenseMatrix m)
        {
            switch (m)
            {
                case GalleryDenseMatrix.Laplacian1D_8: return arena.fProxyLaplacian1D(8);
                case GalleryDenseMatrix.MinIJ_5:        return arena.fProxyMinIJ(5);
                case GalleryDenseMatrix.Pei5_2:          return arena.fProxyPei(5, (fProxy)2);
                case GalleryDenseMatrix.Hilbert4:        return arena.fProxyHilbert(4);
                case GalleryDenseMatrix.Pascal5:         return arena.fProxyPascal(5);
                case GalleryDenseMatrix.Lehmer5:         return arena.fProxyLehmer(5);
                case GalleryDenseMatrix.Fiedler5:  return arena.fProxyFiedler(5);
                case GalleryDenseMatrix.Clement4:  return arena.fProxyClement(4);
                case GalleryDenseMatrix.Rosser8:   return arena.fProxyRosser();
                case GalleryDenseMatrix.DenseNonsym20:   return DenseNonsym(ref arena, 20, 0x51D01u);
                case GalleryDenseMatrix.ConvDiffDense40: return ConvDiffDense(ref arena, 40);
                case GalleryDenseMatrix.Grcar8:          return arena.fProxyGrcar(8);
                case GalleryDenseMatrix.Lauchli3_05:  return arena.fProxyLauchli(3, (fProxy)0.5);
                case GalleryDenseMatrix.Lauchli3_1e3: return arena.fProxyLauchli(3, (fProxy)1E-3);
                // ... WideRandom10x30 / RankDeficient20x10_Rank5 / RandSPD{Well,Ill}Cond20 via
                // arena.fProxyMat(...) + Rand.withRankInPlace / Rand.spdInPlace / Rand.conditionedInPlace
                default: throw new System.ArgumentException("fProxyKrylovBatteryGallery.Build: unhandled GalleryDenseMatrix");
            }
        }

        public static fProxyBSR Build(ref Arena arena, GalleryBSRMatrix m)
        {
            switch (m)
            {
                case GalleryBSRMatrix.Poisson2D_20x20:       return Poisson2D(ref arena, 20, 20);
                case GalleryBSRMatrix.Laplacian2D_16x16:     return arena.fProxyLaplacian2D(16, 16);
                case GalleryBSRMatrix.RandomSparseSPD_120_2: return arena.fProxyRandomSparseSPD(120, 2, (fProxy)0.2, 0x5EED0u);
                case GalleryBSRMatrix.RandomSparseNonsym_80: return arena.fProxyRandomSparse(80, 80, 1, (fProxy)0.1, 0x5EED1u);
                default: throw new System.ArgumentException("fProxyKrylovBatteryGallery.Build: unhandled GalleryBSRMatrix");
            }
        }

        // DenseNonsym / ConvDiffDense / Poisson2D: same shape as the equivalent private helpers
        // already duplicated in IDRTests.fProxy.cs / PreconditionerBatteryTests.fProxy.cs --
        // consolidate here as the ONE copy every battery + migrated special file shares.
    }
}
```

This is also the natural home for the shared `DenseToBSR1x1` conversion helper (currently
duplicated in `BlockCGTests.fProxy.cs` and elsewhere) if a check needs to run a Sparse-tagged
solver over a matrix that only exists in dense literature form.

## 3. Solver-invoker interfaces

New (test-only) file:
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovBattery.Invokers.fProxy.cs`.
These interfaces are **test infrastructure**, not a public library feature -- they live under
`TemplateSourceTests` (compiles into `SourceTests/Generated`, never ships in the UPM package),
not `TemplateSource`. This mirrors the existing `IfProxyPredicate` struct-functor idiom
(interface method generic over the CALLING context, concrete non-generic-over-itself struct
implementations) with one deliberate extension: because a single invoker instance must drive
BOTH the dense literature gallery (`fProxyDenseOperator`) and the BSR gallery
(`fProxyBSROperator`), and must run BOTH the unpreconditioned and identity-explicit paths for
the identity-fold check, its `Solve`/`SolveWithPrecond` methods are themselves generic (over
`TOp`, and `TPre` where applicable) -- the same `<TOp,TPre>` shape every production solver
entry point (`Krylov.cg<TOp,TPre>`, `Krylov.gmres<TOp,TPre>`, ...) already uses. This is a
**new-for-this-codebase** generic shape (generic method ON an interface implemented by a
struct that is itself a type parameter of the battery job) -- see SS9's spike requirement
before fanning out to all solvers.

```csharp
namespace LinearAlgebra
{
    public interface IfProxySquareSolverInvoker
    {
        MatrixProfile Requires { get; }
        MatrixProfile Forbids { get; }
        PreconditionerKind PrecondKind { get; }   // for the Sparse-only preconditioned check
        fProxy Tol { get; }
        int MaxIter(int n);

        /// Allocate/resize any caller-owned scratch vectors for an n x n system. No-op for
        /// solvers whose production entry point self-allocates from Allocator.Temp (gmres,
        /// fgmres, idr). Called once per gallery matrix, before any Solve* call.
        void Init(ref Arena arena, int n);

        SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator;

        SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner;
    }

    public interface IfProxyBlockSolverInvoker
    {
        MatrixProfile Requires { get; }
        MatrixProfile Forbids { get; }
        PreconditionerKind PrecondKind { get; }
        fProxy Tol { get; }
        int MaxIter(int n);

        /// True for solvers whose Requires includes Nonsymmetric: the dense gallery path must
        /// wrap A in fProxyDenseOperatorGeneral, NOT fProxyDenseOperator (see SS4's landmine).
        /// BSR entries are unaffected (fProxyBSROperator.ApplyBlock -> BSR.spMM is general).
        bool NeedsGeneralDenseOperator { get; }

        void Init(ref Arena arena, int n, int s);   // s = block width (RHS count)

        BlockSolveInfo Solve<TOp>(in TOp A, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator;

        BlockSolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyMxN B, ref fProxyMxN X)
            where TOp : struct, IfProxyLinearOperator
            where TPre : struct, IfProxyPreconditioner;

        /// The scalar solver this block family reduces to at s=1 / is compared per-column
        /// against (SS5.3 #6). E.g. BcgInvoker.ScalarCounterpart() returns a CgInvoker.
        IfProxySquareSolverInvoker ScalarCounterpart();
    }

    public interface IfProxyLstsqSolverInvoker
    {
        MatrixProfile Requires { get; }
        MatrixProfile Forbids { get; }
        fProxy Tol { get; }
        int MaxIter(int rows, int cols);

        void Init(ref Arena arena, int rows, int cols);

        /// damp: 0 for the plain-solve checks; the damped-path check (SS5.4 #12) calls this a
        /// second time with damp > 0. No TPre-generic overload: lsqr/lsmr have no
        /// IfProxyPreconditioner-generic entry point in production (their only "preconditioning"
        /// is column (Jacobi) scaling via the lsqrJacobi/lsmrJacobi convenience wrappers or the
        /// fProxyColScaledOperator<TInner> wrapper) -- a Jacobi-scaled variant is wired as a
        /// SEPARATE invoker implementing this same interface (LsqrJacobiInvoker), not a second
        /// method here. There is consequently no identity-fold check for this family (SS5.4).
        LstsqInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x, fProxy damp)
            where TOp : struct, IfProxyLinearOperator;
    }
}
```

## 4. The dense-operator-kind landmine (block family only)

`fProxyDenseOperator.ApplyBlock` computes `Vrows*A` -- documented correct **only when
`A = A^T`**. Nonsymmetric block solvers already avoid this: `Krylov.bbiCGStab`/`Krylov.bgmres`'s
dense convenience overloads wrap A in `fProxyDenseOperatorGeneral` instead (see the doc comment
on `bgmres(in fProxyMxN A, ...)`: *"fProxyDenseOperator's ApplyBlock is symmetric-only and would
silently solve A^Tx=b here"*). The battery must reproduce this choice per invoker
(`NeedsGeneralDenseOperator`), not assume `fProxyDenseOperator` uniformly:

- `bcg`, `bcgrq`, `bfbcg`, `bminres` -> `Requires` includes `SPD`/`SymmetricIndefinite` only ->
  `NeedsGeneralDenseOperator = false` -> dense gallery entries wrap in `fProxyDenseOperator`.
- `bbiCGStab`, `bgmres` -> `Requires` includes `Nonsymmetric` -> `NeedsGeneralDenseOperator =
  true` -> dense gallery entries wrap in `fProxyDenseOperatorGeneral`.
- BSR gallery entries always use `fProxyBSROperator` regardless (`BSR.spMM` has no
  symmetric-only shortcut).

Scalar (non-block) solvers are unaffected -- `IfProxyLinearOperator.Apply`/`ApplyT` are already
general on both wrappers; only `ApplyBlock` differs.

## 5. Battery jobs and standard checks

### 5.1 Job shape (recommended default)

One **concrete** (non-generic) `[BurstCompile] IJob` per family, matching the
`SolverBatteryTests`/`PreconditionerBatteryTests` precedent exactly, with an outer `SolverKind`
enum selecting which concrete invoker struct to build, and ONE shared **generic private
method** (`RunStandardChecks<TInvoker>`) that runs the whole standard-checks battery across
every applicable gallery matrix:

```csharp
// Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovSquareBatteryTests.fProxy.cs
public class fProxyKrylovSquareBatteryTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum SolverKind { Cg, Fcg, Minres, MinresQLP, BiCGStab, Gmres, Fgmres, Idr }
        public SolverKind Kind;
        public NativeArray<fProxy> Fail;   // [0] flag [1] matrix-enum-as-int [2] check-id [3] got [4] expected

        public void Execute()
        {
            switch (Kind)
            {
                case SolverKind.Cg:        RunStandardChecks(new CgInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.Fcg:       RunStandardChecks(new FcgInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.Minres:    RunStandardChecks(new MinresInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.MinresQLP: RunStandardChecks(new MinresQLPInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.BiCGStab:  RunStandardChecks(new BiCGStabInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20 }); break;
                case SolverKind.Gmres:     RunStandardChecks(new GmresInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 4, Restart = 30 }); break;
                case SolverKind.Fgmres:    RunStandardChecks(new FgmresInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 4, Restart = 30 }); break;
                case SolverKind.Idr:       RunStandardChecks(new IdrInvoker { TolValue = Consts.fProxySqrtEps, MaxIterMul = 20, S = 4, Seed = 0x9E3779B1u }); break;
            }
        }

        void RunStandardChecks<TInvoker>(TInvoker inv) where TInvoker : struct, IfProxySquareSolverInvoker
        {
            foreach (GalleryDenseMatrix gm in System.Enum.GetValues(typeof(GalleryDenseMatrix)))
                if (Applicable(inv.Requires, inv.Forbids, GalleryProfiles.Of(gm)))
                    CheckDense(inv, gm);

            foreach (GalleryBSRMatrix gm in System.Enum.GetValues(typeof(GalleryBSRMatrix)))
                if (Applicable(inv.Requires, inv.Forbids, GalleryProfiles.Of(gm)))
                    CheckBSR(inv, gm);
        }
        // CheckDense/CheckBSR run checks #1-#5 from SS5.2 and record into Fail on first failure.
    }

    static System.Array GetKinds() => System.Enum.GetValues(typeof(TestJob.SolverKind));

    [TestCaseSource(nameof(GetKinds))]
    public void SquareBattery(TestJob.SolverKind kind)
    {
        var fail = new NativeArray<fProxy>(5, Allocator.TempJob);
        try
        {
            new TestJob { Kind = kind, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{kind}: matrix={fail[1]} check={fail[2]} got={fail[3]} expected={fail[4]}");
        }
        finally { fail.Dispose(); }
    }
}
```

`System.Enum.GetValues` inside `Execute()`/`RunStandardChecks` runs on the **main thread**
inside `.Run()` (not `.Schedule()`), same constraint every existing battery already accepts --
confirm this compiles under `[BurstCompile] CompileSynchronously = true`; if `Enum.GetValues`
turns out not to be Burst-legal (it may not be -- it's reflection), replace it with a small
`const int` count + a `(GalleryDenseMatrix)i` cast loop, or a hand-written `static
GalleryDenseMatrix[] All => new[] { ... }` array literal built once outside the job and passed
in as a `NativeArray<int>`. Resolve this exact mechanism during the SS9 spike, not by guessing
here -- it is a Burst-legality question, not a design question.

Block and least-squares battery files (`KrylovBlockBatteryTests.fProxy.cs`,
`KrylovLstsqBatteryTests.fProxy.cs`) follow the identical shape: one `SolverKind` enum, one
`RunStandardChecks<TInvoker>` generic method, one `[TestCaseSource]` entry point.

**Alternative** (finer NUnit granularity): make the job itself generic
(`TestJob<TInvoker> : IJob where TInvoker : struct, IfProxySquareSolverInvoker`), closed-
instantiated once per solver, with `[TestCaseSource]` yielding `(SolverKind, GalleryMatrix)`
pairs so each gallery matrix is its own visible NUnit result. Functionally equivalent; not
recommended as the default because it is an unprecedented Burst-job shape in this codebase
(no existing generic `IJob<T>` struct) layered on top of the already-new generic-interface-
method shape -- stacking two novel Burst-generics risks together for a reporting-granularity
win only. Use it if the SS9 spike shows the concrete-job design has some other problem the
generic-job design doesn't.

### 5.2 Square-family standard checks

Each check below runs once per `(invoker, applicable gallery matrix)` pair found by
`RunStandardChecks`. `n` = matrix size, `A` = the wrapped operator
(`fProxyDenseOperator`/`fProxyBSROperator`), `TolBand(matrix)` = a small lookup keyed off
`WellConditioned`/`IllConditioned` (mirroring `SolverBatteryTests`'s hardcoded per-matrix
bands -- NOT a live `Analysis.cond()` call, to keep the battery cheap): e.g.
`WellConditioned -> 50 * Consts.fProxySqrtEps`, `IllConditioned -> 5E-2`.

1. **Converges.** Draw a random `b` (fixed seed per matrix), `x0 = 0`, run
   `inv.Solve(in A, in b, ref x)`. Assert `info.Solved` (or `MaxIterations` with a residual
   still inside the check-tolerance -- see the `Degenerate`/`MaxIterations` note in SS6) and
   assert the FRESH (recomputed, not solver-reported) relative residual
   `||b - A x|| / ||b|| <= 10 * inv.Tol`. Reuse the `RelResidualDense`/`RelResidualBSR` shape
   already duplicated in `IDRTests.fProxy.cs`/`PreconditionerBatteryTests.fProxy.cs` -- move
   the ONE shared copy into `fProxyKrylovBatteryGallery` or a small
   `fProxyKrylovBatteryOracles` helper class alongside it.

2. **Correctness vs. direct-solve reference.** Same `(A, b)` as #1. Compute a reference `xRef`
   via the library's own direct solver for that matrix's KIND: `LU.decompSolve` for
   `Nonsymmetric`/`SymmetricIndefinite`, `CHO.decompSolve` for `SPD`. Assert
   `|x[i] - xRef[i]| <= TolBand(matrix) * (1 + |xRef[i]|)` elementwise.

3. **Determinism.** Run `inv.Solve` twice from independently-allocated `x0 = 0` on the
   identical `(A, b)`. Assert `x1[i] == x2[i]` bit-for-bit for every `i`, and
   `info1.iterations == info2.iterations`.

4. **Identity-fold.** Run `inv.Solve(in A, in b, ref xA)` (the production unpreconditioned
   convenience path) and `inv.SolveWithPrecond(in A, default(fProxyIdentityPreconditioner), in
   b, ref xB)` (the generic path with an explicit identity) from identical `x0`. Assert
   `xA[i] == xB[i]` bit-for-bit and `infoA.iterations == infoB.iterations` -- this directly
   regression-guards the `IsIdentity` compile-time fold every solver's doc comment promises.

5. **Preconditioned convergence** -- only when the current gallery matrix carries `Sparse`.
   Build `M` per `inv.PrecondKind` (`SymmetricBSR -> arena.fProxyBlockJacobi(in A)`,
   `NonsymmetricBSR -> arena.fProxyILU0(in A)`), run `inv.SolveWithPrecond(in A, in M, in b, ref
   x)`, assert the same residual bound as #1.

### 5.3 Block-family additions (on top of #1-#5, block-shaped)

6. **Per-column matches scalar.** For each row `j` of `B`, run `inv.ScalarCounterpart()`'s
   `Solve` on that row as an independent RHS; assert `X[j,:]` matches the scalar `x` elementwise
   within `TolBand`.

7. **Block advantage.** `blockInfo.iterations <= max over j of the scalar solve's iterations`.

8. **Identical-RHS-columns invariant.** Force two rows of `B` bit-identical (e.g. `B[1] =
   B[3]`), solve, and assert `X[1,:]` matches `X[3,:]` within `TolBand` (NOT required to be
   bit-identical -- `BlockCGTests.fProxy.cs`'s existing `RankDeficientBlockDeflates` uses the
   same tolerance-bounded form, not bit-identity, for this same invariant; match that
   precedent). This is the check that would have caught the historical `bminres` bug (SS0) --
   make it unconditional for every block invoker, not opt-in.

9. **Rank-deficient RHS graceful.** Same forced-duplicate-rows `B` as #8. Assert no `NaN`/`Inf`
   anywhere in `X`, and `info.status` is not a hard-crash outcome (Converged or MaxIterations
   with finite output; never Breakdown producing garbage -- if a genuine Breakdown is possible
   here for some family, decide via the SS9 spike whether that is itself the correct contract
   and encode it, don't silently loosen the assertion).

### 5.4 Least-squares-family checks (its own smaller list -- different shape from #1-#5)

10. **Overdetermined -> min-residual.** On an `Overdetermined | FullRank` matrix: solve, then
    call the production `Krylov.lstsqResidual<TOp>(in A, in b, in x, damp: 0, ...)` oracle
    (already exists -- `OP/Krylov.Lstsq.Common.fProxy.cs`, recomputes `rnorm`/`Arnorm`/`xnorm`
    fresh from `x`, no need to reimplement). Assert `Arnorm` is small relative to `||A|| * ||x||`
    (the optimality residual, not the plain residual, since an overdetermined system usually
    has nonzero residual even at the optimum).

11. **Underdetermined -> min-norm.** On an `Underdetermined | FullRank` matrix: solve via the
    invoker, and independently via `SVD.pinvSolve` (already exists,
    `OP/SVD.Solvers.fProxy.cs`). Assert the two solutions match elementwise within `TolBand`
    (both should converge to the unique minimum-norm solution).

12. **Damped path.** Run with `damp = 0` and confirm it is bit-identical to the plain solve
    (already a documented production guarantee -- `"damp == 0 is BIT-IDENTICAL to the plain
    solve"` on both `lsqr`/`lsmr`'s doc comments; this is a regression guard on that promise,
    not new behavior). Then run with `damp > 0` and assert
    `lstsqResidual(..., damp, ...).Arnorm` (the damped optimality residual) is near zero.

There is no identity-fold check for this family (SS3's `IfProxyLstsqSolverInvoker` doc note) --
Jacobi column-scaled variants (`lsqrJacobi`/`lsmrJacobi`) are wired as separate invokers
(`LsqrJacobiInvoker`, `LsmrJacobiInvoker`) implementing the same interface, exercised through
the SAME #10-#12 checks, not a bespoke 4th method.

## 6. Convergence-status nuance

Several existing checks in the bespoke files assert `info.status ==
IterativeSolveStatus.Converged` strictly. The battery's `IllConditioned`-tagged matrices may
legitimately need more iterations than a cheap `MaxIterMul * n` budget affords. Two options,
pick one during implementation and apply it consistently (do not mix per-solver):

(a) Size `MaxIterMul` generously enough (per the existing bespoke files' own budgets, e.g.
`20 * n` for cg/idr, `4 * n` total across restarts for gmres) that every `WellConditioned` AND
`IllConditioned` gallery entry converges within budget -- keep the "Converges" check's `Solved`
assertion strict.

(b) Accept `MaxIterations` as a pass for check #1 PROVIDED the fresh residual bound still holds
(the solver got "close enough" even if it didn't cross the exact `Solved` threshold) -- do NOT
accept `Breakdown` or `Degenerate` as a pass anywhere in the standard battery; those are always
failures for a well-posed (full-rank, no deliberately-singular) gallery entry.

## 7. Special-case boundary (stays OUT of the battery)

Keep in small residual files (one per solver family, or fold into a single
`Krylov{Square,Block,Lstsq}SpecialCasesTests.fProxy.cs` per family if that reads cleaner --
implementer's call):

- **GMRES/FGMRES**: restart-count sensitivity (`restart=1` degenerate, `restart>=n` exact-in-
  one-cycle), and any solver-vs-solver cross-check that doesn't generalize (`MatchesGmres` in
  `IDRTests.fProxy.cs` -- IDR-specific, not every solver has a natural sibling reference).
- **IDR(s)**: `s`-parameter edge (`s=1` degenerate shadow-space width -- `SEqualsOne` in
  `IDRTests.fProxy.cs`), and reproducibility under the DEFAULT (omitted) seed specifically
  (`DeterminismDefaultSeed`) -- the battery's own determinism check (#3) already covers
  same-explicit-seed reproducibility if the invoker's `Solve` always threads a fixed seed, so
  only the zero-arg-convenience-overload path is genuinely residual.
- **MINRES-QLP**: min-length solution behavior on a deliberately SINGULAR symmetric system
  (this needs a `Singular`-tagged gallery entry the standard battery doesn't include, since
  every family's #1/#2 checks assume a well-posed full-rank system) -- this is new coverage,
  not migrated from anywhere (SS0).
- **GCRO-DR** (when it ships): restart-recycling / deflation-subspace behavior across a
  SEQUENCE of related solves (changing A and/or b) -- inherently outside a single-solve battery
  check.
- **BCGrQ/BFBCG**: any column-dropping/deflation-width-specific behavior beyond the generic
  rank-deficient-RHS check (#9) -- review `BlockCGrQTests.fProxy.cs`/`BlockBFBCGTests.fProxy.cs`
  during migration (SS8) to identify what's genuinely special vs. already covered.
- **KrylovVerifyAtExitTests.fProxy.cs / KrylovFusedKernelTests.fProxy.cs**: not reviewed in
  detail for this spec -- inventory these during migration; from their names they likely test
  solver-internals (the "verify claimed Converged with one fresh residual" behavior noted in
  `Krylov.fProxy.cs`'s `MakeSolveInfo` doc comment, and fused-kernel numerics) rather than
  matrix-regime coverage, so they probably stay special almost entirely.

## 8. Migration plan

### 8.1 Current solver inventory (verified, SS0)

| Family | Solvers | Existing bespoke test file(s) |
|---|---|---|
| Square | cg, fcg, minres, minresQLP, biCGStab, gmres, fgmres, idr | `ConjugateGradientTests`, `FlexibleCGTests`, `KrylovPMinresTests`, (none -- new), `GMRESTests`, `FGMRESTests`, `IDRTests` |
| Block | bcg, bcgrq, bfbcg, bbiCGStab, bgmres, bminres | `BlockCGTests`, `BlockCGrQTests`, `BlockBFBCGTests`, `BlockBiCGStabTests`, `BlockGmresTests`, `BlockMinresTests` |
| Least-squares | lsqr, lsmr (+ Jacobi-scaled variants) | scattered across `MultiRHSSolveTests`, `SparseSolverTests`, `KrylovRound2Tests` (not fully inventoried for this spec -- first migration step below covers that) |

### 8.2 Sequence (battery green before deleting anything)

1. **Infra only.** Land `KrylovBatteryProfile.cs` (singular), `KrylovBattery.Gallery.fProxy.cs`,
   `KrylovBattery.Invokers.fProxy.cs` with invoker structs for every solver in SS8.1's table.
   No existing file touched, nothing deleted.
2. **Spike one solver per family** (`cg`, `bcg`, `lsqr`) end-to-end: invoker + battery job +
   `[TestCaseSource]` entry, confirm it compiles under `[BurstCompile]` and is green. This is
   where the two novel-for-this-codebase generic shapes (SS3's generic interface method,
   SS5.1's `Enum.GetValues`-inside-`Execute()` question) get resolved empirically before the
   remaining ~13 solvers are wired the same way. Do not touch any bespoke file yet.
3. **Fan out** the rest of SS8.1's inventory into invokers + battery coverage, INCLUDING the
   from-scratch MINRES-QLP coverage (SS0, SS7) and the `Singular`-tagged gallery entry it needs.
   Full test suite green with the battery ADDED (nothing removed yet).
4. **Fold bespoke files one at a time.** For each file in SS8.1's table: diff its scenarios
   against the battery's standard checks (SS5.2-5.4); delete only the now-duplicated scenarios;
   keep genuinely special ones (SS7) in a slimmed residual file. Run the full suite green after
   EACH file's edit, so a regression is attributable to one diff. Order doesn't matter much;
   suggest starting with the files with the LEAST special content (`ConjugateGradientTests`,
   `BlockCGTests`) to validate the fold pattern cheaply before the more special-heavy ones
   (`IDRTests`, `GMRESTests`).
5. **DEVLOG + cleanup.** One dated DEVLOG entry (per `CLAUDE.md`) under a `## Krylov test
   battery` heading summarizing the consolidation (not per-file). Sweep `docs/features/*.md`
   for any stale references to deleted test-file names (there should not be any -- public docs
   are instructed not to name test classes per `CLAUDE.md`, but check anyway).

### 8.3 Explicitly out of scope for this migration

TFQMR, GCRO-DR (not implemented -- SS0), CRAIG/CRAIGMR (removed -- SS0). When any of these ship,
wiring them into the battery is exactly the SS8.2-step-3-shaped addition SS9's "how a new
solver plugs in" estimates.

## 9. How a new solver plugs in

Concretely, for a hypothetical new square solver `Krylov.foo<TOp,TPre>(A, M, b, ref x, ref
scratch1, ref scratch2, maxIter, tol)` with a `foo<TOp>` unpreconditioned convenience overload
(the standard shape every existing solver already has):

```csharp
public struct FooInvoker : IfProxySquareSolverInvoker
{
    public fProxy TolValue; public int MaxIterMul;
    fProxyN scratch1, scratch2;

    public MatrixProfile Requires => MatrixProfile.SPD;      // or whatever foo actually needs
    public MatrixProfile Forbids => MatrixProfile.None;
    public PreconditionerKind PrecondKind => PreconditionerKind.SymmetricBSR;
    public fProxy Tol => TolValue;
    public int MaxIter(int n) => MaxIterMul * n;

    public void Init(ref Arena arena, int n)
    {
        scratch1 = arena.fProxyVec(n);
        scratch2 = arena.fProxyVec(n);
    }

    public SolveInfo Solve<TOp>(in TOp A, in fProxyN b, ref fProxyN x) where TOp : struct, IfProxyLinearOperator
        => Krylov.foo(in A, in b, ref x, ref scratch1, ref scratch2, MaxIter(A.Rows), Tol);

    public SolveInfo SolveWithPrecond<TOp, TPre>(in TOp A, in TPre M, in fProxyN b, ref fProxyN x)
        where TOp : struct, IfProxyLinearOperator where TPre : struct, IfProxyPreconditioner
        => Krylov.foo(in A, in M, in b, ref x, ref scratch1, ref scratch2, MaxIter(A.Rows), Tol);
}
```

Plus one `SolverKind.Foo` enum entry and one `case` line in the battery job's `Execute()`
switch. That is roughly 20 lines total (the exact count scales with how many scratch buffers
the production solver needs -- `minresQLP` needs 9, `cg` needs 3) versus a 230-340-line bespoke
file -- a >85% reduction even in the worst (most-scratch-buffers) case.

Worked concrete examples using solvers that already exist:

- **`Krylov.idr`**: `IdrInvoker` needs no `Init` scratch (idr self-allocates from
  `Allocator.Temp`, like `gmres`/`fgmres`); stores `S` and `Seed` as fields; `Requires =
  Square` (works on any square kind -- the dense-nonsymmetric literature entries just exercise
  it the hardest); `PrecondKind = NonsymmetricBSR`. `Solve<TOp>` forwards to
  `Krylov.idr(in A, in b, ref x, S, MaxIter(A.Rows), Tol, Seed)`; `SolveWithPrecond<TOp,TPre>`
  to `Krylov.idr(in A, in M, in b, ref x, S, MaxIter(A.Rows), Tol, Seed)`.
- **`Krylov.bcg`**: `BcgInvoker` owns `R, P, Q, Z : fProxyMxN` scratch, sized `(s, n)` in
  `Init(ref arena, n, s)`; `Requires = SPD`, `NeedsGeneralDenseOperator = false`,
  `ScalarCounterpart() => new CgInvoker { TolValue = TolValue, MaxIterMul = MaxIterMul }`.
  `Solve<TOp>` forwards to `Krylov.bcg(in A, in B, ref X, ref R, ref P, ref Q, MaxIter(A.Rows),
  Tol)`; `SolveWithPrecond<TOp,TPre>` to `Krylov.bcg(in A, in M, in B, ref X, ref R, ref P, ref
  Q, ref Z, MaxIter(A.Rows), Tol)`.

## 10. Constraints (recap)

- Templates are the source of truth; `fProxy` token -> codegen emits float/double partials.
  Never hand-edit `Assets/LinearAlgebra/Source*/Generated/**` or `SourceTests/Generated/**`.
- `[BurstCompile] IJob` via `.Run()`; struct-functor generics only (no managed delegates/LINQ
  in any Burst-compiled path, including inside the battery jobs themselves).
- `Assert.IsTrue(bool)` / `Assert.AreEqual` ONLY inside Burst-compiled code -- an interpolated-
  string `Assert.IsTrue(cond, $"...")` overload is BC1071 and silently forces a Mono fallback
  (see `OP/DEVLOG.md`'s `BlockCGTests` entry and the burst-test-compile-gotchas memory note).
  The `Fail : NativeArray<fProxy>` diagnostic payload exists precisely so the RICH failure
  message (built with string interpolation) can be assembled OUTSIDE the Burst job, in the
  plain `[Test]` method, after `.Run()` returns.
- Per-precision tolerances scale with `Consts.fProxySqrtEps` via the `IsDouble()` idiom
  (`(double)Consts.fProxyEpsilon < 1e-10`), matching `SolverBatteryTests`/
  `PreconditionerBatteryTests` exactly -- do not hardcode a single tolerance shared by both
  float and double.
- Gallery access via `Arena` + `using LinearAlgebra.Gallery;` (opt-in) for the literature
  matrices; `Rand.*InPlace` (no opt-in needed, plain `LinearAlgebra` namespace) for the
  synthetic well/ill-conditioned and rank-deficient entries.
- Comments state contracts only; rationale/history goes to the folder's `DEVLOG.md`, never
  inline (per `CLAUDE.md`).
- Green gate: exact `Result=Passed total=N passed=N failed=0` from the headless suite runner
  (`Tools/*.ps1`) -- no `| tail`, no partial-output reads.

## 11. Open questions for the owner / test-writer to resolve empirically (not guesses)

1. Is `System.Enum.GetValues` actually Burst-legal inside `[BurstCompile] Execute()`? (SS5.1.)
   `SolverBatteryTests`/`PreconditionerBatteryTests` never call it FROM INSIDE the job -- they
   call it in the outer `GetEnums()` static method, which runs on the managed/NUnit side, not
   inside `Execute()`. The design above needs it inside `RunStandardChecks` (called from
   `Execute()`) to loop gallery matrices generically per invoker. If it doesn't compile,
   fall back to a hand-written `static readonly GalleryDenseMatrix[] All` array (built as a
   `const int` count + index cast, still avoiding a hardcoded per-solver matrix list).
2. Does the doubly-generic interface-method shape (SS3) actually compile and run correctly
   under Burst for a NON-trivial case (e.g. `bminres`'s scratch-heavy signature)? Resolve via
   the SS8.2-step-2 spike before fanning out.
3. Exact `MaxIterMul`/`Restart` budgets per solver -- pick values so every `WellConditioned` AND
   `IllConditioned` gallery entry converges (SS6 option (a)), OR adopt option (b) and skip this
   tuning. Owner/test-writer's call; both are internally consistent, just pick one.
