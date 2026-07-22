# Spec: fix for Krylov.bminres (block MINRES) s>1 divergence

## 1. The bug, precisely

**File:** `reference/wip-bminres/Krylov.Block.MINRES.fProxy.cs`

**Wrong line:** `BuildOmega` (private helper), the `Y` assembly:

```csharp
static bool BuildOmega(in fProxyMxN Gbar, in fProxyMxN Beta, ref fProxyMxN Omega, ref fProxyMxN Gamma, int s)
{
    int s2 = 2 * s;
    var Y = new fProxyMxN(s2, s, Allocator.Temp, true);
    CopyRowsAt(in Gbar, ref Y, 0, s, s);
    CopyRowsAt(in Beta, ref Y, s, s, s);   // <-- BUG: Beta copied UNtransposed
    ...
```
(line numbers in the WIP file: the `CopyRowsAt(in Beta, ...)` call is at line 103; the
call site that supplies the wrong `Beta` argument is at line 335: `if (!BuildOmega(in
Gbar, in Beta, ref OmegaNew, ref Gamma, s))`.)

**Fix:** `Y`'s bottom `s` rows must be `Beta^T`, not `Beta`. Either transpose at the
call site into a fresh scratch buffer, or (recommended, no extra allocation) transpose
inline while writing `Y`:

```csharp
static bool BuildOmega(in fProxyMxN Gbar, in fProxyMxN Beta, ref fProxyMxN Omega, ref fProxyMxN Gamma, int s)
{
    int s2 = 2 * s;
    var Y = new fProxyMxN(s2, s, Allocator.Temp, true);
    CopyRowsAt(in Gbar, ref Y, 0, s, s);
    for (int r = 0; r < s; r++)
        for (int c = 0; c < s; c++)
            Y[s + r, c] = Beta[c, r];      // Beta^T, not Beta
    ...
```

Nothing else in the file needs to change. In particular, do **not** touch:
- `M2`'s bottom-right assembly (`M2[s + r, s + c] = Beta[r, c]`, line 324) -- this
  correctly uses `Beta` **untransposed**. It is a different physical role from
  `BuildOmega`'s `Y` (see SS3) and must stay as-is.
- `Blas.dot(in OmegaOld, in M2, ref Result, true, false)` (Omega-apply direction) --
  correct as-is.
- `Blas.dot(in OmegaNew, in PhibarStack, ref Res2, true, false)` (Phibar propagation)
  -- correct as-is.
- `BlockCTV(in OldEps, in W1, ...)` / `BlockCTV(in Delta, in W2, ...)` (search-direction
  terms) -- correct as-is.
- `LU.solveInPlaceTransA(ref GammaCopy, ref pivS, ref W)` (the `Gamma^T` solve; one of
  the two prior coder attempts, per the task brief) -- correct, **keep it**. It was a
  necessary fix but not sufficient on its own; this `BuildOmega` fix was the missing
  second piece.
- `BlockCTV(in Phi, in W, ref T)` (X update) -- correct as-is.
- `Phibar := Transpose(Beta)` at setup (SS4.4) -- correct as-is.
- SS4.1's Lanczos step (`BlockCTV(in Beta, in Vprev, ref T)`, i.e. `Beta^T . Vprev`) --
  correct as-is; **do not** touch it (verified independently, see SS2 below).

This is a **single-site, single-transpose fix**. Every other transpose/orientation
choice in the file was checked exhaustively (SS4 below) and is already correct.

## 2. Why this specific line, and why it was invisible at s=1

`Beta` (the WIP's block-Lanczos normalization factor, produced by
`BlockNormalizeIdentity`/`BlockNormalizePrecond` each iteration) is lower-triangular
and, for `s > 1`, **not symmetric** -- `Beta != Beta^T` in general. `BuildOmega`'s `Y`
argument and `M2`'s bottom-right both currently receive the *same* raw `Beta` value,
but they need *different* orientations of it:

- `M2`'s bottom-right feeds `Result = OmegaOld^T . M2`, which extracts `Epsln`(next)
  and `Dbar`(next) -- both derived through the *old* rotation. This role needs `Beta`
  **untransposed** (confirmed by exhaustive search, SS4).
- `BuildOmega`'s `Y = [Gbar; Beta_arg]` is QR-factored **directly**, with no `OmegaOld`
  involved, to produce `Gamma`/`Omega` for *this* iteration. This role needs `Beta^T`
  (proven directly: `Y`'s bottom block must equal the true block-tridiagonal
  subdiagonal entry `T[c+1,c] = Q_(c+1)^T . A . Q_c`, which is numerically confirmed to
  equal `Beta^T`, not `Beta` -- see SS3).

At `s = 1`, `Beta` is a 1x1 matrix, so `Beta == Beta^T` trivially -- the bug has
*zero* numerical effect, which is exactly why `MatchesScalarAtS1` passes while every
`s > 1` test fails. This also explains the task's "smoking gun": two RHS rows that are
bytewise identical (`B[1] == B[3]`) feed the *same* block-Lanczos subspace, but the
`Gbar`/`Gamma`/`Omega` bookkeeping built from the wrong (asymmetric) `Y` breaks the
block-symmetric structure that would otherwise force `X[1] == X[3]` -- hence the
observed row-symmetry break.

## 3. Proof: the true block-tridiagonal subdiagonal entry is Beta^T, not Beta

Numerically verified (`reference/wip-bminres/bminres_reference.py`, function
`check_lanczos_orthogonality` + the `run_lanczos`/`build_T` pair) on an n=10, s=2
symmetric indefinite `A`:

```
k=1: true Q_(k+1)^T A Q_k =
[[ 9.122989 -1.04783 ]
 [-0.        9.634771]]
      Betas[k]  (raw, WIP Beta_ours_k) =
[[ 9.122989  0.      ]
 [-1.04783   9.634771]]
      Betas[k]^T =
[[ 9.122989 -1.04783 ]
 [ 0.        9.634771]]
      match raw? False   match transposed? True
```
(same result at k=2, k=3 -- see the script's Lanczos/T-matrix construction).

Here `Q_k = V_k^T` is the block-Lanczos vector in the classical column-major sense
(`V_k` is the WIP's row-major `s x n` block, row = vector). `Q_(k+1)^T . A . Q_k` is
the textbook, convention-independent definition of the block-tridiagonal matrix's
subdiagonal entry -- and it equals `Beta_ours_k^T`, never `Beta_ours_k`.

## 4. Proof of the fix: exhaustive numerical search + before/after trace

**Method** (full detail and runnable code in `reference/wip-bminres/bminres_reference.py`):

1. Built a **ground-truth** solver (`bruteforce_X`) that is completely independent of
   the Omega/Gamma/Delta/Epsln/Phibar recursion: it runs the (separately verified,
   `check_lanczos_orthogonality`) block-Lanczos recurrence for `m` steps, assembles the
   explicit `(m+1)s x ms` block-tridiagonal matrix `T` from the collected
   `Alfa`/`Beta` blocks, and solves the projected least-squares problem
   `min || Beta_0^T e_1 - T.Y ||` directly via `numpy.linalg.lstsq`. Verified this
   converges to `A^-1 . B` to ~1e-16 by `m = 5` on an n=10 system (see script output,
   SS5).
2. Ported the WIP's SS4.2/4.3 recursive algorithm line-for-line into Python
   (`bminres_ref`), preserving the WIP's exact row-major operations (`BlockGram`,
   `BlockCTV` = `C^T . V`, `Blas.dot(A, B, transA, transB)`, the `M2`/`Result`
   extraction, `BuildOmega`, the `Gamma^T` solve).
3. Ran an **exhaustive grid search** (512 = 2^9 combinations) over every transpose
   choice in the suspect region: `M2`'s Beta orientation, `BuildOmega`'s `Y`-bottom
   Beta orientation (**decoupled from #1** -- this was the key move once the tied
   single-flag search failed), the `Omega`-apply direction for `M2` and for
   `PhibarStack` (decoupled), the `OldEps`/`Delta` search-direction term orientations,
   the `Gamma` solve orientation, the `X`-update orientation, and the `Phibar`-init
   orientation. Correctness metric: max error vs `A^-1.B` **and** the
   identical-RHS-rows-give-identical-X-rows invariant, across 5 random
   `(n, s, seed)` trials per combination.
4. **Exactly one combination reached machine precision** (`err = 1.5e-9` on the 5-trial
   sweep, tightening to `1e-16`/`1e-17` with a larger iteration budget): every flag
   matching the current WIP source **except** `BuildOmega`'s `Y`-bottom argument, which
   needed the transpose. All other 511 combinations, including the "transpose both
   Beta uses together" combination that seemed like the natural first guess, left a
   residual error of `>= 1e-3` on at least one trial.
5. Re-ran the fix across 40 additional random `(n, s)` trials (`s` from 1 to 5, `n`
   from 6 to 19) plus three larger cases (`n=30,s=5`; `n=20,s=6`; `n=16,s=2`) with a
   generous iteration budget:

```
worst error over 40 random (n,s) trials, WITH fix    = 1.7e-09   (limited by the 6n iteration budget)
worst error over 40 random (n,s) trials, WITHOUT fix = 2.499e+00 (current WIP; diverges)
n=30 s=5: max abs err with fix, maxIter=10n = 5.6e-17
n=20 s=6: max abs err with fix, maxIter=10n = 6.9e-17
n=16 s=2: max abs err with fix, maxIter=10n = 4.9e-17
s=1 fixed == unfixed: True   (confirms the fix is a no-op at s=1, MatchesScalarAtS1 unaffected)
```

The identical-RHS-rows invariant (`B[2] == B[0]` forced in every trial with `s >= 3`)
is included in the "worst error" metric above and holds to the same precision -- i.e.
**the `X[1] != X[3]` symmetry break described in the task is confirmed reproduced by
the unfixed reference, and confirmed resolved by the fix.**

## 5. Minimal reproduction (for the coder to sanity-check after applying the fix)

`n = 10`, `s = 2`, symmetric indefinite `A` (seed 3, `BuildDenseSymIndefinite`-style:
random symmetric + a `+/-n` diagonal split), `B` random (seed 4). Run
`reference/wip-bminres/bminres_reference.py` directly:

```
python reference/wip-bminres/bminres_reference.py
```

Expected output (already reproduced and committed):

```
=== (a) ground-truth solver self-check, and Lanczos orthogonality ===
Lanczos orthogonality residual (should be ~1e-15): 2.65e-15
  bruteforce m=2   max|X-Xtrue| = 1.8e-02
  bruteforce m=5   max|X-Xtrue| = 1.8e-16
  bruteforce m=10  max|X-Xtrue| = 1.4e-16

=== (b)/(c) recursive solver: bug reproduction (fix=False) vs fix (fix=True) ===
  worst error over 40 random (n,s) trials, WITH fix    = 1.7e-09
  worst error over 40 random (n,s) trials, WITHOUT fix = 2.5e+00  (current WIP)

=== s=1 sanity: fix is a no-op at s=1, MatchesScalarAtS1 unaffected ===
  s=1 fixed == unfixed: True
```

For a smaller, hand-inspectable trace of the exact iteration-by-iteration values
(`Gamma_1`, `Delta_2`, `Gbar_2`, ...) that pin down the bug, see SS3's `k=1` block
above and the script's `build_omega`/`bminres_ref` functions with `fix=False` vs
`fix=True` -- both are directly runnable and print identical intermediate shapes,
differing only in the `omega_arg = BetaNext.T if fix else BetaNext` line.

## 6. What the coder should do

1. Apply the one-line fix in SS1 to `BuildOmega` in
   `reference/wip-bminres/Krylov.Block.MINRES.fProxy.cs`.
2. Update `BuildOmega`'s doc comment (currently: "...of the thin-QR left factor Qy of
   the 2s x s stack `[Gbar; Beta]`...") to say `[Gbar; Beta^T]`, since that is now
   accurate.
3. Move the WIP file and its test file out of `reference/wip-bminres/` (gitignored)
   into the real template location
   (`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.Block.MINRES.fProxy.cs` and
   `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/BlockMinresTests.fProxy.cs`,
   per `docs/dev/spec-bminres.md`'s SS10 file path), regenerate
   (`Tools/regen.ps1`), and run the 7 `BlockMinresTests` (float + double). All 5
   previously-failing `s > 1` tests are expected to pass; `MatchesScalarAtS1` and
   `IdentityFoldBitIdentical` are expected to keep passing unchanged.
4. Run the full project test suite headlessly and confirm
   `Result=Passed total=N passed=N failed=0`.
5. Add a dated "Krylov.bminres" DEVLOG.md entry (per CLAUDE.md) recording: the bug
   (`BuildOmega`'s `Y`-bottom needed `Beta^T`, not `Beta`), that it was diagnosed via a
   numerical reference + exhaustive grid search (not hand-derivation -- two prior
   hand-derivation attempts, including the `Gamma`-transpose fix that turned out to be
   correct-but-insufficient, did not find it), and a pointer to
   `reference/wip-bminres/bminres_reference.py` for anyone re-deriving this in future.
6. Once green, proceed with `docs/dev/spec-bminres.md`'s SS11 checklist items 6-7
   (README/DEVLOG) that were presumably blocked on this fix.

## 7. Out of scope

- No other file in the WIP needs changes (SS1's "do not touch" list is exhaustive per
  the grid search in SS4).
- Do not touch scalar `Krylov.minres` (`OP/Krylov.fProxy.cs`) -- unaffected, and the
  s=1 reduction test already confirms it is not implicated.
- Do not touch `BlockGram`, `BlockCTV`, `BlockAdd`, `CopyBlock`, `CopyMat`,
  `BlockApplyPre`, `CountConverged` (`Krylov.Block.Common.fProxy.cs`) -- shared with
  `bcg`/`bcgrq`, unmodified, not implicated by this diagnosis.
- Do not re-attempt a from-scratch re-derivation of SS4.2/4.3's math; the fix in SS1 is
  complete and verified. If a *future* regression reintroduces a similar bug, re-run
  `reference/wip-bminres/bminres_reference.py`'s grid-search methodology (SS4) rather
  than hand-deriving -- it is what actually found this bug after two hand-derivation
  attempts failed.
- Preconditioned `bminres` (`BlockNormalizePrecond` path): not separately re-verified
  numerically here (the numpy reference only models the identity-preconditioner path),
  but per `docs/dev/spec-bminres.md` SS5, `BuildOmega` and the whole SS4.2/4.3
  recursion are identical under preconditioning (M only touches `BlockNormalize`) --
  the same fix is expected to resolve `PreconditionedMatchesScalar` too. The coder
  should confirm this empirically via the existing test (SS6 step 3), not by
  extending the numpy reference.

## 8. STATUS UPDATE (post-application): fix confirmed NECESSARY but NOT SUFFICIENT

The `BuildOmega` `Beta^T` fix from SS1 was applied to the real C#
(`reference/wip-bminres/Krylov.Block.MINRES.fProxy.cs` -- `Y[s + r, c] = Beta[c, r]` at
lines 103-105, matching SS1 exactly) and the project regenerated. Result: the **same 5
`s > 1` tests fail identically** (`BlockAdvantageIterations`, `KnownSolutionRecovered`,
`MatchesScalarMinresPerColumn`, `PreconditionedMatchesScalar`,
`RankDeficientBlockNoNaN`); `MatchesScalarAtS1` and `IdentityFoldBitIdentical` still
pass. The numpy port in SS4/SS5 converges to machine precision with this fix; the real
C# does not. **A second, distinct bug remains**, located outside the 9 transpose flags
grid-searched in SS4.

### Re-verification performed (before concluding a second bug exists)

To rule out "the numpy port just isn't faithful enough to reproduce the real bug"
before escalating:

1. Re-read `reference/wip-bminres/Krylov.Block.MINRES.fProxy.cs`'s current `BuildOmega`
   (post-fix) line by line against `bminres_reference.py`'s `build_omega`. They already
   matched exactly, including the `Qperp` seed-and-project construction (the `[0;I]`/
   `[I;0]` seeds, the `Qy^T.Z0` / `Qy.(Qy^T.Z0)` projection, the `QR.decomp(Z1)` ->
   `Qperp`/`Rz` step, the `Rz` diagonal rank check, the `[Qy | Qperp]` assembly) -- this
   was **not** a "clean full-QR shortcut" in the committed script; it was already a
   line-for-line port of the seed-and-project method.
2. Re-read `OP/OP.Dot.fProxy.cs`'s `dot(in fProxyMxN a, in fProxyMxN b, ref fProxyMxN c,
   bool transposeA = false)` (the overload `Blas.dot(..., true, false)` /
   `Blas.dot(..., false, false)` actually route through, since `transposeB = false` in
   both `BuildOmega` calls) to confirm its `m`/`n`/`k` contract is plain
   `c = a^T . b` (or `a . b`) with no hidden convention -- confirmed, matches the numpy
   port's `Qy.T @ Z0` / `Qy @ T` exactly.
3. Ran a **gauge-freedom check**: replaced the seed-and-project `Qperp` construction
   with a completely different, independent method (`numpy.linalg.qr(Y, mode='complete')`,
   taking the full orthogonal factor's trailing `s` columns directly, no seed/projection
   at all) and re-ran the same 30-trial random sweep. Both methods converge to the same
   machine-precision result (`8.774e-11` worst error, identical), and an added
   per-iteration invariant check (`Omega^T.Omega == I`, `(Omega^T.[Gbar;Beta^T])[s:,:] ==
   0`) never fired across all 30 trials x ~6n iterations each. This confirms: (a) the
   seed-and-project method, as coded/ported, is not itself numerically unstable or
   convention-sensitive -- gauge freedom genuinely holds -- and (b) the numpy port's
   `build_omega` is not "accidentally correct via a different, non-faithful shortcut."

None of this re-verification found a further discrepancy. Per this document's own
methodology (SS4: numerical reference over hand-derivation), continuing to hand-compare
C# against the numpy port line-by-line has now been tried twice and found nothing
further -- the same failure mode this whole investigation exists to avoid.

### Next step (blocking): instrumented C# dump

The definitive way to close the remaining gap is to dump the **real C#'s** actual
per-iteration intermediates for a small, fixed-seed `s = 2` (or `s = 3`) case matching
one of the 5 still-failing tests, and diff them directly against
`bminres_reference.py`'s ground truth for the *same* `A`/`B`. Needed per iteration
`k = 0, 1, 2, ...`:

- `Alfa`, `Beta` (post-`BlockNormalizeIdentity`)
- `Gbar`, `Dbar` (both the value fed into `BuildOmega` and the value carried to the
  next iteration)
- `Delta`, `Epsln`, `Gamma`
- `Omega` (or at least the two invariant checks: `max|Omega^T.Omega - I|` and
  `max|(Omega^T.[Gbar;Beta^T])[s:, :]|` -- if either is non-negligible, the bug is
  inside `BuildOmega`/`QR.decomp` itself, not the recursion around it)
- `Phi`, `Phibar`
- `W` (post `LU.solveInPlaceTransA`)
- `X` after the update

A temporary `Debug.Log`/test-only dump inside the `bminres` loop (guarded by an
iteration count and `s == 2`) for one of the currently-failing tests'
`A`/`B` (or a small custom fixed-seed matrix matching `bminres_reference.py`'s
`make_indef`/`rng.uniform` construction, so the same numbers can be fed to both sides)
is sufficient. Once obtained, diff against `bminres_reference.py`'s `bminres_ref`
(same `A`, `B`, `fix=True`) iteration by iteration; the first iteration where a
value diverges pinpoints the remaining bug.

Candidate spots **not yet independently verified** against the real C# (all currently
taken on faith from the doc comment / code read, not from an actual behavioral check) that
the instrumented dump should specifically confirm or rule out:
- `LU.solveInPlaceTransA`'s actual solved system (`Gamma^T . W = RHS` vs some other
  convention) -- never independently verified beyond trusting its name/doc comment.
- `Blas.trans` (used once, at `Phibar` setup) -- trivial, low suspicion, but unverified.
- `CopyRowsAt` / `CopyBlockAt` -- re-read and believed correct (SS1's "do not touch"
  list), but not verified via an isolated unit trace.
- Whether `QR.decomp`'s `genHouseholder` zero-column fallback (`OP/QR.fProxy.cs:30-53`,
  sets `u[k] = sqrt(2)` when the column norm is below `zeroThreshold`) produces a
  genuinely different -- not just differently-gauged -- result than LAPACK/numpy's own
  zero-column handling in some edge case triggered by this specific recursion. Low
  suspicion (the 4 non-`RankDeficientBlockNoNaN` failing tests use well-conditioned
  random matrices with no deliberate rank deficiency), but not ruled out.
