# Survey: square, non-CG Krylov solver landscape (minres / biCGStab / gmres)

Status: RESEARCH / SURVEY (no solver code here). Same lens as the BFBCG replacement of ad-hoc ridge
block-CG (`reference/papers/BFBCG-algorithm-extract.md`): for each SQUARE non-CG solver we ship, is
there a better-named, better-behaved modern method that should replace or sit alongside it. This
document is a map for future single-session specs, not an implementation spec -- signatures below are
scoping sketches, not final APIs. `cg`/`fcg` are mentioned only for contrast (already the
most-worked-on solver in the family; not re-audited here).

## 1. Inventory: what we already ship

All four scalar square solvers live in `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/` and share one
shape: a single generic `<TOp, TPre>` body per method, with an unpreconditioned `<TOp>` entry point that
forwards into it using `fProxyIdentityPreconditioner`. The `IsIdentity` field on that preconditioner is
a compile-time-literal branch that Burst constant-folds per specialization, so the unpreconditioned path
compiles to, and is bit-identical to, the classic textbook recurrence with zero preconditioner traffic.
This is a completed 2026-07-19 refactor (`OP/DEVLOG.md:44-80`, "SCALAR KRYLOV FAMILY NOW FULLY
COLLAPSED") that deleted the formerly-separate `pminres`/`pbiCGStab`/`pgmres` explicit-scratch bodies
(39+ call sites swept) -- there is now exactly one loop per method, not two.

### 1.1 minres -- symmetric (possibly indefinite) Ax = b
`Krylov.fProxy.cs:526-712` (doc `506-525`). Paige-Saunders (1975) Lanczos tridiagonalization plus an
incrementally Givens-rotated QR of the tridiagonal system, tracking the running scalar `phibar` as the
(possibly M-weighted) residual norm for free -- no extra dot/matvec to test convergence
(`OP/DEVLOG.md:1773`). Requires A symmetric (not necessarily definite); a real preconditioner M must be
SPD (caller-verified, not checked beyond NaN-safe breakdown guards). Left-preconditioned in the
Lanczos inner product (standard symmetric preconditioning, not GMRES-style right-preconditioning).
Eight scratch vectors (`y,r1,r2,v,w,w1,w2,z`); `z` unused/`default` under identity. Stopping:
`phibar^2 <= tol^2 * ||b||^2` under identity (exact, no verify); under a real M, `phibar` is the
M^-1-weighted residual, so a claimed Converged/MaxIterations exit is checked against one FRESH
`||b-Ax||` before being trusted (`Krylov.fProxy.cs:680-696, 703-711`) -- only Breakdown reports the
unverified value. History: `pminres` (2026-07-18, `OP/DEVLOG.md:1001-1022`) was the original
preconditioned body; merged into `minres` 2026-07-19 (`OP/DEVLOG.md:66-80`) -- the merge notes the two
bodies genuinely differed (init-check ordering) and had to be reconciled, not just pasted together.
Concrete pairings today: none beyond the generic `<TOp,TPre>` -- no BSR x preconditioner rung ladder
exists for minres the way cg has one (`Krylov.fProxy.cs:289-498`); every symmetric-indefinite
preconditioned solve currently has to call the generic form directly.

### 1.2 biCGStab -- general nonsymmetric square Ax = b
`Krylov.fProxy.cs:811-963` (doc `793-810`). Van der Vorst (1992) short two-sided recurrence: flat O(n)
memory, no growing Krylov basis (unlike gmres). RIGHT-preconditioned (`pHat = M^-1 p`, `sHat = M^-1 s`
drive the A-applies and the x update -- `811-816` doc). Never calls `A.ApplyT`
(`Interfaces/LinearOperator.fProxy.cs:24`, explicitly called out) -- this is the one transpose-free
solver in the square family, which matters for BSR operators that only cheaply support forward `spMV`.
Seven scratch vectors (`r,rHat0,p,v,t,pHat,sHat`); `rHat0` is the FIXED shadow residual (set once to
the initial `r`, `Krylov.fProxy.cs:867`), never updated -- this is the classical (non-randomized) choice
and matters directly for Section 2's IDR(s) discussion below. Three standard breakdown modes detected
(`rho==0`/NaN, `rHat0.v==0`/NaN, `omega==0`/NaN) -- no restart, no look-ahead. History: `pbiCGStab`
(2026-07-18) merged into `biCGStab` 2026-07-19 (`OP/DEVLOG.md:56-64`). Concrete pairings: ILU0, SPAI,
RestrictedSchwarz (RAS) over BSR -- RAS is explicitly biCGStab-ONLY because it is not symmetric and has
no `cg` rung (`Krylov.fProxy.cs:467-469`).

### 1.3 gmres(m) -- general nonsymmetric square Ax = b, restarted
`Krylov.GMRES.fProxy.cs:32-185` (doc `11-31`). Arnoldi + modified Gram-Schmidt orthonormal basis,
incrementally Givens-rotated least-squares over that basis, hard restart every `restart` (`m`) inner
steps to bound memory (all state -- basis `V`, Hessenberg `H`, Givens `cs/sn`, rotated rhs `g`,
solution `y`, work vector `w` -- allocated from `Allocator.Temp` per call, unlike the flat-scratch cg/
biCGStab/minres primitives: `Krylov.GMRES.fProxy.cs:57-66`, doc `27`). RIGHT-preconditioned (runs GMRES
on `A.M^-1`; Arnoldi residual stays the true `||b-Ax||` so the convergence test is unchanged --
`14-25` doc). A hard restart with NO deflation: every restart throws away the Krylov subspace and starts
over from the current `x`, which is the textbook GMRES(m) stagnation failure mode on problems with a few
troublesome small/complex eigenvalues. Only one preconditioner pairing exists today: BSR x ILU0
(`Krylov.GMRES.fProxy.cs:215-221`) -- there is no BSR x AMG pairing (contrast `fProxyAMGPreconditioner`,
which wires cg AND fcg but not gmres: `MG/fProxyAMGPreconditioner.fProxy.cs:37-91`), and no flexible
variant exists, so gmres cannot correctly pair with a variable preconditioner at all (a K-cycle AMG
explicitly REJECTS plain `cg` with an `ArgumentException` and demands `fcg` --
`MG/fProxyAMGPreconditioner.fProxy.cs:40-41,48` -- gmres has no such escape hatch because there is
nothing to escape to). History: `gmres`/`pgmres` (2026-07-18, `OP/DEVLOG.md:134-148`) merged into one
body 2026-07-19 (`OP/DEVLOG.md:44-53`), completing the scalar-family collapse.

### 1.4 For contrast: cg / fcg (SPD only, not re-audited here)
`cg<TOp,TPre>` (`Krylov.fProxy.cs:128-255`) is the most heavily worked solver in the file -- seven
BSR-preconditioner rungs (BlockJacobi/SSOR/IC0/FSAI/Chebyshev/AdditiveSchwarz/AMG). `fcg`
(`Krylov.FCG.fProxy.cs`, Notay 2000 flexible CG / Polak-Ribiere direction) already exists specifically
to stay valid under a VARYING preconditioner (K-cycle AMG, inner-iterative smoothers) and is the correct
current pairing for `fProxyAMGPreconditioner`'s K-cycle mode. This flexible/K-cycle story exists for the
SPD side and has no nonsymmetric analogue -- see Section 2c (FGMRES).

## 2. Per-solver modern-replacement survey

### 2a. minres -> MINRES-QLP (Choi, Paige & Saunders, SISC 2011 / TOMS 2014)
MINRES-QLP extends Paige-Saunders MINRES with a second, QLP, factorization phase that activates once the
solver detects the underlying tridiagonal system is (numerically) rank-deficient -- it then switches from
the standard QR-based update to a QL-based one that provably returns the MINIMUM-LENGTH solution among
all least-squares-residual minimizers, on both consistent and inconsistent, possibly SINGULAR, symmetric
(indefinite) systems. Plain MINRES on a singular/near-singular symmetric system can converge to an
arbitrary vector in the solution set (any `x* + null(A)` component); MINRES-QLP is the fix. This
directly matters for this library's existing users: LOBPCG's buckling/generalized eigenproblems
(`lobpcg-structural-stability` memory) and any KKT/saddle-point system from the QP path are exactly the
"symmetric, indefinite, possibly (near-)singular" class MINRES-QLP targets, and the shipped LOBPCG
robustness work already had to hand-roll SVQB-style regularization for a RELATED rank-deficiency problem
(`lobpcg-tiny-penalty-collapse` memory) -- MINRES-QLP would be the general-purpose version of that same
fix, usable as an inner solver.

**Verdict: AUGMENT, not replace.** Plain MINRES is the cheaper, simpler, correct choice whenever A is
known nonsingular (or the caller does not need the min-length guarantee) -- it should stay the default.
MINRES-QLP is a strictly more complex two-phase algorithm (extra rotation, phase-transition logic,
rank estimate) for a specific robustness guarantee; ship it as a SEPARATE named solver (`minresQLP`,
analogous to how `lsqr`/`lsmr` coexist rather than one replacing the other) so callers opt in when they
know they may hit singularity, without paying the extra bookkeeping cost on the common well-posed path.

### 2b. biCGStab -> IDR(s), BiCGStab(l), QMR/TFQMR, GCR
- **IDR(s)** (Sonneveld & van Gijzen, SISC 2008): induced dimension reduction -- forces the residual into
  a sequence of shrinking subspaces `G_j` via a fixed "shadow" space `P` (`s` vectors), typically
  faster and more robust than BiCGStab for nonsymmetric systems, especially for `s > 1` (IDR(1) is
  close kin to BiCGStab; IDR(4) is the commonly-cited sweet spot). **Determinism note (required by the
  task brief):** the reference formulation draws `P`'s columns RANDOM (`orth(randn(n,s))`); for this
  library `P` MUST be a FIXED deterministic set, e.g. the first `s` columns of `I` (or a fixed
  structured/Hadamard-style pattern) orthonormalized once via QR at solve start with no RNG in the loop
  -- this is a documented IMPLEMENTATION CONSTRAINT, not a blocker: IDR(s) convergence theory does not
  require P to be random, only that P have full column rank and (empirically) not be pathologically
  aligned with A's invariant subspaces, which a fixed structured basis satisfies for generic problems.
- **BiCGStab(l)** (Sleijpen & Fokkema, ETNA 1993): replaces BiCGStab's single-step stabilization
  polynomial with a degree-`l` GMRES-like minimization every `l` steps -- specifically fixes the
  near-breakdown / stagnation BiCGStab exhibits when A has near-imaginary eigenvalue pairs. Niche
  relative to this library's typical BSR workloads (elliptic-PDE-like, SPD-adjacent operators where cg/
  minres already apply); valuable only if a genuinely oscillatory/convection-dominated use case shows up.
- **QMR / TFQMR** (Freund & Nachtigal 1991 / Freund 1993): TFQMR is transpose-free (like biCGStab -- no
  `ApplyT` needed, so it fits every existing `IfProxyLinearOperator`), quasi-minimizes the residual for a
  visibly SMOOTHER convergence curve than BiCGStab's characteristic sawtooth, and avoids one of
  BiCGStab's two breakdown conditions by construction. Straightforward short-recurrence method,
  structurally close to what is already shipped.
- **GCR** (Eisenstat, Elman & Schultz 1983): flexible (varying-preconditioner-safe) but stores a growing
  set of search directions like unrestarted GMRES -- for this library it is better understood as the
  theory GCRO-DR (Section 2c) builds on than as a standalone addition; no separate GCR ship recommended.

**Verdict: AUGMENT with TFQMR (low risk, transpose-free, smoother curve) now; IDR(s) as the recommended
new nonsymmetric DEFAULT once the fixed-shadow determinism adaptation is implemented and validated;
BiCGStab(l) and GCR deferred (niche / subsumed).** Keep biCGStab itself -- it is the cheapest-memory
nonsymmetric option and the only one with existing ILU0/SPAI/RAS preconditioner pairings; do not retire.

### 2c. gmres(m) -> GCRO-DR / GMRES-DR, LGMRES, FGMRES, pipelined/CA variants
- **FGMRES** (Saad, SISC 1993): lets the preconditioner VARY between inner steps -- store the
  preconditioned basis vector `Z_j = M_j.Apply(v_j)` directly (instead of one `M^-1` apply folded in at
  the solution-accumulation step, as today's right-preconditioned `gmres` does) so `x`'s update sums
  `y_i Z_i` instead of applying `M^-1` once to the accumulated `w`. This is a SMALL structural delta
  from the existing `gmres<TOp,TPre>` body (one extra `n x (m+1)` workspace array `Z`, same Arnoldi loop,
  same Givens rotations) -- it is the exact nonsymmetric mirror of the fcg-vs-cg relationship already
  shipped, and is the ONLY way to correctly pair a K-cycle AMG (or any inner-iterative smoother) with a
  restarted-GMRES outer solve; today that pairing simply does not exist (Section 1.3).
- **GCRO-DR / GMRES-DR** (deflated restarting; Morgan 2002 GMRES-DR, Parks/de Sturler/Mackey/Johnson/
  Maiti SISC 2006 GCRO-DR): recovers exactly the stagnation-across-restarts loss that a hard restart
  throws away, by carrying a small deflation subspace of (harmonic) Ritz vectors approximating A's
  smallest eigenvalues across restarts -- and, in GCRO-DR's specific formulation, across a SEQUENCE of
  related systems (changing A and/or b between solves), which is directly relevant to this library's
  repeated-solve workloads (LQR/MPC resolves, quasi-static/evolving BSR physics scenes). Belos ships
  `BelosGCRODRIter.hpp`/`BelosBlockGCRODRIter.hpp` (BSD-3) as a structural iteration-loop reference.
- **LGMRES** (Baker, Jessup & Manteuffel, SISC 2005): a cheaper alternative augmentation -- carries
  approximate error directions from previous cycles instead of an eigenvalue-targeted deflation
  subspace. Simpler than GCRO-DR, less powerful on genuinely eigenvalue-clustered stagnation; a
  reasonable "if GCRO-DR proves too complex" fallback, not a parallel ship target.
- **Pipelined / communication-avoiding GMRES** (s-step / pipelined variants, e.g. Ghysels & Vanroose
  2014 and similar): explicitly LOW priority here for two independent reasons -- (1) this is a
  single-node library, so there is no communication latency to hide/avoid; (2) these methods typically
  reorder or batch reductions (s-step Krylov bases, delayed dot products) for latency hiding, which is
  in direct tension with `FloatMode.Strict` cross-arch determinism (forbids reassociation of `+-*/sqrt`
  reductions -- `determinism-analysis` memory), so adopting one would cost both engineering effort and
  the library's determinism guarantee for a benefit (multi-node/multi-device latency hiding) this
  project has no deployment target for. Do not pursue.

**Verdict: AUGMENT with FGMRES first (low effort, completes the flexible-preconditioning story gmres is
currently missing entirely), then GCRO-DR (higher effort, high value for the repeated/evolving-system
workloads this library already has). LGMRES only if GCRO-DR's complexity proves not worth it. Keep plain
restarted gmres(m) -- it remains the correct low-memory default for a one-off nonsymmetric solve with a
FIXED preconditioner.**

## 3. Cross-cutting: determinism + preconditioner fit

- **IDR(s)'s shadow space** is the only candidate in this survey whose textbook formulation is
  non-deterministic (random `P`). The fixed deterministic adaptation (Section 2b) is REQUIRED before any
  `idrs` ships; flag this explicitly in that future spec's acceptance criteria (byte-identical results
  across runs / architectures under `FloatMode.Strict`, the same bar every other solver in this file
  meets).
- **MINRES-QLP / FGMRES / GCRO-DR / TFQMR** are all standard sqrt-and-dot-product recurrences structurally
  identical in kind to what is already shipped (no random sampling, no reduction-order tricks) -- they
  fit the existing determinism story with no special adaptation beyond the usual `DetMath`-routed
  transcendentals if any appear (none of these algorithms need more than `sqrt`).
- **Preconditioner-suite pairing** (AMG, Chebyshev, FSAI/SPAI, Additive/Restricted Schwarz -- all
  shipped per the preconditioner-suite work): FGMRES is the natural nonsymmetric pairing for AMG's
  K-cycle (mirrors `fcg` + `fProxyAMGPreconditioner`'s K-cycle rung,
  `MG/fProxyAMGPreconditioner.fProxy.cs:72-91`) -- once FGMRES exists, wiring an AMG K-cycle rung for it
  is small, symmetric-effort work. GCRO-DR pairs well with any FIXED preconditioner (ILU0/SPAI/Schwarz)
  since its deflation subspace is orthogonal to the choice of `M`; no new preconditioner interface is
  implied. MINRES-QLP reuses the same M-SPD contract plain `minres` already has (no new preconditioner
  shape).

## 4. Gap analysis + prioritized roadmap

| # | Candidate | Value | Effort | Notes |
|---|---|---|---|---|
| 1 | **FGMRES** | HIGH | LOW | Completes the flexible-preconditioning story gmres is missing; near-identical structural delta from shipped `gmres<TOp,TPre>` |
| 2 | **GCRO-DR** | HIGH | MEDIUM-HIGH | Deflated restart recovers real stagnation; valuable for this library's repeated/evolving-system solves |
| 3 | **IDR(s)** | HIGH | MEDIUM (+determinism adaptation) | Modern nonsymmetric default; needs the fixed-shadow adaptation validated before shipping |
| 4 | **MINRES-QLP** | MEDIUM-HIGH | MEDIUM-HIGH | Robustness on singular/rank-deficient symmetric-indefinite (KKT, buckling); two-phase QR-to-QL logic is genuinely more intricate than the existing minres merge |
| 5 | **TFQMR** | MEDIUM | LOW-MEDIUM | Cheap augment alongside biCGStab; transpose-free, smoother curve, one fewer breakdown mode |
| 6 | BiCGStab(l) | LOW-MEDIUM | MEDIUM | Niche (near-imaginary spectra); defer until a concrete oscillatory workload appears |
| 7 | LGMRES | LOW-MEDIUM | LOW-MEDIUM | Fallback if GCRO-DR proves too complex; do not ship both |
| 8 | GCR (standalone) | LOW | LOW | Theory absorbed into GCRO-DR; no standalone ship |
| 9 | Pipelined/CA GMRES | LOW | -- | Rejected: no multi-node benefit here, conflicts with `FloatMode.Strict` reduction-order guarantee |

### (a) FGMRES -- recommended first target
Spec sketch: new file `OP/Krylov.FGMRES.fProxy.cs`, `fgmres<TOp,TPre>(in A, in M, in b, ref x, int
restart, int maxIter, fProxy tol)` sharing `gmres`'s Arnoldi/Givens loop exactly, but allocating an
additional `Z` workspace (`m+1` vectors, same `Allocator.Temp` lifetime as `V`/`H`) storing
`Z_j = M_j.Apply(v_j)` per inner step, and accumulating the solution as `x += sum y_i Z_i` (no final
`M.Apply` on the accumulated direction). Ship the identity-forwarding `fgmres<TOp>` for symmetry with
every other solver in the file (even though FGMRES's whole point is a real M -- keep the API shape
consistent; the identity path is then trivially bit-identical to plain gmres, useful as a correctness
cross-check in tests). First concrete pairing: BSR x AMG K-cycle (`fProxyAMGPreconditioner`), mirroring
the existing `fcg` rung. Acceptance-test idea: (1) with a FIXED (non-varying) preconditioner, `fgmres`
output is bit-identical to `gmres` (same M applied every step is the degenerate case both algorithms
agree on); (2) with AMG's K-cycle as M, `fgmres` converges where plain `gmres`+K-cycle would silently
give a wrong/non-converging answer (K-cycle violates the fixed-M assumption `gmres` relies on for its
residual identity) -- this is the same "wrong-without-flexibility" property the AMG preconditioner's
existing `ArgumentException` guard already encodes for cg vs fcg
(`fProxyAMGPreconditioner.fProxy.cs:40-41,48`); mirror that guard on `gmres` + K-cycle once `fgmres`
exists. Does not replace `gmres` -- adds alongside, same relationship as `fcg` to `cg`.

### (b) GCRO-DR
Spec sketch: builds on FGMRES landing first (deflation vectors compose naturally with a flexible
preconditioner, though GCRO-DR itself works with a fixed M too -- sequencing FGMRES first is about
shared workspace/loop-shape reuse, not a hard dependency). Port structure reference:
`reference/square/BelosGCRODRIter.hpp` (iteration-loop shape only, BSD-3) plus the original
Parks/de Sturler/Mackey/Johnson/Maiti SISC 2006 paper for the deflation-subspace math (fetch at
implementation time). Acceptance-test idea: construct A with a few known small/troublesome eigenvalues
(e.g. a shifted/perturbed nonsymmetric test matrix), show restarted `gmres(m)` stagnates or needs
materially more total iterations than `gcrodr(m)` with the same restart budget on repeat solves against
a slightly perturbed A/b (the recycling scenario), and that a degenerate single-solve, zero-deflation
GCRO-DR run matches plain gmres's iteration count within a small tolerance.

### (c) IDR(s)
Spec sketch: new `idrs<TOp,TPre>(in A, in M, in b, ref x, int s, int maxIter, fProxy tol, ...)`; MUST
build the fixed shadow basis `P` deterministically (Section 3) as an explicit, documented step (e.g.
QR-orthonormalize the first `s` columns of `I_n`) with no RNG dependency, stated as a doc-comment
CONTRACT ("P is fixed and deterministic, not the textbook random shadow space"). Port reference:
`reference/square/idrs.jl` (MIT, IterativeSolvers.jl -- algorithm cross-check only, never copied
structurally per this project's port-fidelity rule, since the source uses column-major/1-indexed Julia
idiom foreign to this codebase's row-major C# shape). Acceptance-test idea: byte-identical `x`/iteration
count across repeated runs (determinism harness), and a head-to-head iteration-count comparison against
`biCGStab` on a nonsymmetric BSR gallery matrix, expecting IDR(s) to match or beat it per the published
literature's typical finding.

### (d) MINRES-QLP
Spec sketch: new `minresQLP<TOp,TPre>`, SEPARATE from `minres` (Section 2a verdict). Needs its own
scratch-vector set (Paige-Saunders MINRES's `y,r1,r2,v,w,w1,w2` plus the QLP phase's extra rotation
state) and a documented rank/singularity heuristic for the QR-to-QL phase switch. Primary reference:
`reference/square/MINRESQLP-SISC-2011.pdf` (Choi, Paige & Saunders) for the derivation;
`reference/square/minresQLP.py` (Apache-2.0 mirror, algorithm cross-check only) for the reference
recurrence shape. Acceptance-test idea: a symmetric SINGULAR consistent system (rank-deficient A, b in
range(A)) where plain `minres` converges to an arbitrary member of the solution set but `minresQLP`
converges to the documented MINIMUM-NORM solution (verify via `||x||` against a known min-norm
reference, e.g. computed through the pseudoinverse route already shipped for SVD/COD); a symmetric
INDEFINITE well-posed system where both solvers agree (regression: QLP must not regress the common
case).

### (e) TFQMR
Spec sketch: `tfqmr<TOp,TPre>`, transpose-free short recurrence, same scratch-vector budget class as
biCGStab (no `ApplyT`). Reference: `reference/square/BelosTFQMRIter.hpp` /
`BelosPseudoBlockTFQMRIter.hpp` (BSD-3, iteration-loop shape) plus Freund's 1993 SIAM J. Sci. Comput.
paper for the quasi-minimization math. Acceptance-test idea: on a nonsymmetric BSR test matrix where
biCGStab exhibits its characteristic residual-norm sawtooth (near-breakdown wobble), show tfqmr's
tracked residual is monotonically non-increasing (or close to it) while reaching the same final
tolerance in a comparable iteration count. Adds alongside biCGStab -- does not replace it.

## 5. Reference implementations + licensing

Ports must come from PERMISSIVE sources (algorithms are not copyrightable; always reimplement from the
math, never copy source) -- the library is heading to public v1.0 with an existing GPL-taint cleanup
item, so this bar is stricter than "any license." Fetched into `reference/square/` (gitignored):

| File | Method(s) | License | Use |
|---|---|---|---|
| `BelosGCRODRIter.hpp`, `BelosBlockGCRODRIter.hpp` | GCRO-DR | BSD-3 (Trilinos) | iteration-loop structure reference |
| `BelosTFQMRIter.hpp`, `BelosPseudoBlockTFQMRIter.hpp` | TFQMR | BSD-3 (Trilinos) | iteration-loop structure reference |
| `BelosRCGIter.hpp` | Recycling CG (adjacent to GCRO-DR's recycling idea, SPD side; not itself on this survey's target list) | BSD-3 (Trilinos) | optional cross-read only, not a ship target here |
| `LICENSE-Trilinos.txt` | -- | BSD-3 | license text for the above |
| `MINRESQLP-SISC-2011.pdf` | MINRES-QLP | open-access SIAM journal PDF | primary algorithm derivation (Choi, Paige & Saunders 2011) |
| `minresQLP.py`, `MINRES-QLP-LICENSE.txt` | MINRES-QLP | Apache-2.0 (`github.com/syangliu/MINRES-QLP` mirror of the Stanford SOL code) | reference recurrence shape, cross-check only |
| `idrs.jl`, `LICENSE-IterativeSolvers.jl.txt` | IDR(s) | MIT (`JuliaLinearAlgebra/IterativeSolvers.jl`) | reference recurrence shape, cross-check only (Julia idiom -- never copy structurally) |
| `reference/belos/BelosBlockFGmresIter.hpp`, `BelosPseudoBlockGmresIter.hpp` (already present, prior session) | FGMRES | BSD-3 (Trilinos) | iteration-loop structure reference |

Not fetched (cite at implementation time; low incremental value now or already covered above):
- BiCGStab(l): Sleijpen & Fokkema, ETNA 1, 1993, "BiCGstab(l) for linear equations involving unsymmetric
  matrices with complex spectrum" -- open-access ETNA PDF, fetch only if this method is actually
  scheduled (Section 4 defers it).
- LGMRES: Baker, Jessup & Manteuffel, SIAM J. Sci. Comput. 27(5), 2005 -- check institutional access at
  implementation time; SciPy's `scipy.sparse.linalg.lgmres` (BSD-3) is a permissive implementation
  reference if the paper alone is insufficient.
- GMRES-DR (Morgan 2002) / original GCRO-DR paper (Parks, de Sturler, Mackey, Johnson & Maiti, SIAM J.
  Sci. Comput. 28(5), 2006): fetch the paper alongside the already-fetched Belos header when 4(a) is
  actually implemented -- the header alone gives loop shape, not the deflation-subspace derivation.
- Krylov.jl (`JuliaSmoothOptimizers/Krylov.jl`, MPL-2.0): has Julia implementations of most of the above
  (`gmres.jl`, `fgmres.jl`, `bicgstab.jl`, `minres_qlp.jl`, etc.) -- ALGORITHM cross-check only per this
  project's existing MPL-2.0 policy (`spec-rectangular-krylov-survey.md` Section 5), never read
  side-by-side while writing a port.

## 6. Project constraints (recap for the implementing session)

- Templates under `Assets/LinearAlgebra/CodeGen/TemplateSource*` are the source of truth; never edit
  `Assets/LinearAlgebra/Source/Generated` directly.
- `fProxy` is the codegen dtype token (float/double/int per generation); comments state contracts only
  (what a member requires/destroys/returns) -- algorithm rationale, derivation notes, and rejected
  alternatives go in the folder's `DEVLOG.md`, never in code comments.
- Short parameter names throughout: `maxIter`, `tol` (library-wide ruling -- never expand to
  `maxIterations`/`tolerance`).
- A future BLOCK version of any solver here carries a `b` prefix (`bminres`, `bgmres`, etc., matching
  the shipped `bbicgstab`/`bgmres`/`bminres`/`bcgrq` naming) -- that is a separate track, out of scope
  for the scalar solvers surveyed here.
- Zero-alloc discipline matches the existing family: a generic `<TOp,TPre>` core takes caller-provided
  scratch vectors (or `Allocator.Temp` workspace for the GMRES-shaped ones, matching `gmres` itself);
  convenience overloads allocate once from the Arena.
- Do not touch `README.md` (per this survey's brief).
- This document is a survey: no solver code, no template edits, deliverable is this file.

## Recommended first target

**FGMRES.** Lowest effort of anything in Section 4 (a documented, small structural delta from the
already-shipped `gmres<TOp,TPre>` body -- store the preconditioned basis per step instead of one final
`M.Apply`), yet it is the highest-leverage single change: it is the ONLY item that completes a story this
library has already half-built and shipped for the SPD side (`fcg` + AMG K-cycle,
`fProxyAMGPreconditioner.fProxy.cs:72-91`) but left entirely absent for the nonsymmetric side (`gmres`
has zero AMG pairing today, Section 1.3). Everything else on the list (GCRO-DR, IDR(s), MINRES-QLP,
TFQMR) is a genuinely new recurrence with its own derivation risk; FGMRES is closest to "the coding agent
copies gmres's loop and changes one line's worth of bookkeeping," making it both the safest and the most
strategically overdue next square-solver addition.
