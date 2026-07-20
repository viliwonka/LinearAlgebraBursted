# Spec: eliminate silent false-Converged in single-RHS Krylov solvers (task #53)

Status: DRAFT -- read-only investigation, no code changed. Target for implementation:
`Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Krylov.{CG,MINRES,MINRESQLP,BiCGStab,GMRES,FGMRES,IDR}.fProxy.cs`.
Regenerate via `Tools/regen.ps1` after editing; never hand-edit `Assets/LinearAlgebra/Source/Generated/**`.

## 1. Question

Can `Krylov.{cg,minres,minresQLP,biCGStab,gmres,fgmres,idr}` return `SolveInfo.status ==
Converged` (`Solved == true`) while the true relative residual `||b-Ax||/||b||` is actually large?

**Answer: yes, for 5 of the 7 -- minres (identity/plain path), minresQLP, biCGStab, gmres,
fgmres, idr. cg is already immune. minresQLP's version of this bug is not hypothetical: it is
already observed and documented in production (see §3.2).**

## 2. Verdict table

| Solver | Quantity tested against tol | Estimate or true residual? | Silent false-Converged possible? | Already gated? |
|---|---|---|---|---|
| `cg` | `rr = dot(r,r)` on a directly-updated `r` (`Krylov.CG.fProxy.cs:207`) | direct-vector recurrence (can drift) | **No** -- verified below | **Yes**, verify-at-exit at `Krylov.CG.fProxy.cs:208-218` |
| `minres`, identity (M=I) path | `phibar` (Givens-recurrence scalar) | recurrence estimate | **Yes** | **No** -- explicitly "no verify needed" |
| `minres`, preconditioned (M!=I) path | `phibar` (M^-1-weighted) | recurrence estimate | No | **Yes**, `Krylov.MINRES.fProxy.cs:194-204` |
| `minresQLP` | internal `flag` from `relres`/`relAresl` (scale-normalized) | recurrence estimate, different normalization than the library convention | **Yes -- empirically observed on Rosser** | Recomputes a fresh residual (`Krylov.MINRESQLP.fProxy.cs:373-377`) but **never uses it to gate `status`** |
| `biCGStab` | `ss`/`rr` via `Blas.axpyNormSq` on a directly-updated `r` | direct-vector recurrence (`r` never recomputed from `b-Ax` after init) | **Yes** | **No** -- no verify anywhere in the file |
| `gmres` | `resnorm = abs(g[j+1])` (Arnoldi/Givens least-squares residual) | recurrence estimate, computed **before `x` is even updated** | **Yes** (most severe: decision precedes the vulnerable back-substitution step) | **No** |
| `fgmres` | same as `gmres` | same | **Yes**, identical mechanism | **No** |
| `idr` | `rr` via `Blas.axpyNormSq` on a directly-updated `R` | direct-vector recurrence (`R` never recomputed from `b-Ax` after init) | **Yes** | **No** |

Not in the requested 7, but sharing the same `Forbids=IllConditioned` stopgap (see §5) --
**not analyzed in depth here, flagged for a follow-up**:
- `Krylov.GCRODR.fProxy.cs` -- doc comment at line 40 already claims "never a false Converged";
  has a dedicated collapsed-Hessenberg-pivot Breakdown guard the plain `gmres` lacks. Worth an
  independent audit but likely already immune.
- `Krylov.TFQMR.fProxy.cs` -- doc comment at line 32 states the tracked `tau` is a *proven upper
  bound* on the true residual (`||r|| <= tau*sqrt(halfsteps)`), not a driftable estimate -- a
  structurally different (and safer) guarantee than the Lanczos/Arnoldi recurrences above.
  Likely already immune, but not verified here.

## 3. Per-solver mechanism and code path

### 3.1 `cg` -- NOT vulnerable (reference pattern)

`Krylov.CG.fProxy.cs:207` updates `x`/`r` via `Blas.updateXR` (direct vector recurrence, can
drift over many iterations same as any other solver here), but the very next block already
verifies before trusting it:

```
Krylov.CG.fProxy.cs:208-218
rr = Blas.updateXR(alpha, p, ref x, Ap, ref r);
if (rr <= threshold)
{
    A.Apply(in x, ref Ap);
    r.CopyFrom(in b);
    r.addScaledInPlace((fProxy)(-1), Ap);
    rr = Blas.dot(r, r);
    if (rr <= threshold)
        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));
}
```

On a failed verify it does not return -- it falls through with `r`/`rr` refreshed to the
true values, so the next iteration's `beta`/`p` update is computed from honest data. This is
the template the fixes below all copy. `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovVerifyAtExitTests.fProxy.cs`
already regression-tests this exact behavior (`VerifyAtExitCatchesOptimisticDriftOnIllConditionedMoler`,
`VerifyAtExitAddsExactlyOneApplyAndPreservesSolutionOnHealthySolve`).

### 3.2 `minres` -- vulnerable on the identity (plain) path only

`Krylov.MINRES.fProxy.cs:171-176`:
```
fProxy gamma = math.sqrt(gbar * gbar + beta * beta);
gamma = math.max(gamma, gammaFloor);     // gammaFloor = Consts.fProxyEpsilon, line 121
cs = gbar / gamma;
sn = beta / gamma;
fProxy phi = cs * phibar;
phibar = sn * phibar;
```
and line 184:
```
Blas.combine3(ref w, v, -oldeps, w1, -delta, w2, 1 / gamma);   // w = (v - oldeps*w1 - delta*w2) / gamma
```
`gammaFloor` prevents an exact divide-by-zero, but once the true `sqrt(gbar^2+beta^2)` is below
`gammaFloor`, flooring `gamma` upward makes `cs^2 + sn^2 = (gbar^2+beta^2)/gamma^2 < 1` -- the
"rotation" the recurrence relies on is no longer unitary. That single event simultaneously (a)
inflates `1/gamma` at line 184, amplifying `w` and hence `x` by orders of magnitude -- this is
the exact mechanism `Assets/LinearAlgebra/CodeGen/TemplateSourceTests/DEVLOG.md:126-129` names
as the root cause of Rosser's 1e14-1e19 divergence ("minres's `w = (...)/gamma`... a
small-but-not-exactly-zero pivot... amplifies the update by orders of magnitude") -- and (b)
decouples `phibar`'s decay from the true residual norm, since the Givens step that is supposed
to guarantee `phibar == ||b-Ax||` no longer preserves that invariant. Both effects share one root
cause; nothing stops `phibar` from happening to read *small* on the corrupted iteration instead
of large.

The claimed-convergence check that trusts this unconditionally:
```
Krylov.MINRES.fProxy.cs:188-192
if (phibar * phibar <= threshold)
{
    if (M.IsIdentity)
        // phibar IS ||b-Ax|| at this step (MINRES identity) -- no verify needed.
        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, phibar);
    ...
```
The `M.IsIdentity` branch immediately below (lines 194-204) is the fix already written for the
*preconditioned* case (added 2026-07-18 per `Krylov.PMinres.fProxy.cs`'s original DEVLOG entry,
now merged into this file) -- it recomputes a fresh `r = b-Ax` and only returns `Converged` if
`trueRR <= threshold`, else falls through. The identity path never got the same treatment
because the identity `phibar == ||b-Ax||` claim is an *exact-arithmetic* identity that the
`gammaFloor` clamp breaks in floating point -- the same class of drift the non-identity path was
already patched for.

### 3.3 `minresQLP` -- vulnerable, and the one case with direct evidence

`Krylov.MINRESQLP.fProxy.cs:337-355` derives `rnorm = phi`, `relres = rnorm/(Anorm*xnorm+beta1)`,
and `relAresl`, then sets an internal `flag`:
```
if (relAresl <= tol) flag = 2;
if (relres <= tol) flag = 1;
```
These are the algorithm's own **scale-normalized** stopping tests -- not `||b-Ax||/||b||`. After the
loop, `Krylov.MINRESQLP.fProxy.cs:373-377` unconditionally recomputes a fresh true residual:
```
A.Apply(in x, ref r3);
r1.CopyFrom(in b);
r1.addScaledInPlace((fProxy)(-1), r3);
fProxy finalRnorm = math.sqrt(Blas.dot(r1, r1));
```
but the status mapping right after it (lines 379-382) derives `status` purely from `flag`,
**never comparing `finalRnorm` back against `tol`**:
```
if (flag == 8) status = IterativeSolveStatus.MaxIterations;
else if (flag == 6 || flag == 7 || flag == 9 || flag == -3) status = IterativeSolveStatus.Breakdown;
else status = IterativeSolveStatus.Converged;   // includes flag == 1 and flag == 2
```
So `SolveInfo.rnorm` reported to the caller is honest (it IS `finalRnorm`), but
`SolveInfo.status`/`Solved` can be `Converged` while that same honest `rnorm` is far above
`tol*||b||`. This is not hypothetical -- it is already documented as observed behavior:
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovBattery.Invokers.fProxy.cs:207-219`
(the `fProxyMinresQLPInvoker` doc comment) and
`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/DEVLOG.md:142-149` both record that on the
gallery's Rosser matrix minresQLP's own `relres` criterion "exits" (i.e. sets `flag=1`, mapped
to `Converged`) with the fresh residual landing "~13-14x looser... independent of the absolute tol
requested" than the library's `||b-Ax||/||b||` convention, and on Rosser specifically "its own relres
criterion still exits well short of this battery's fresh-residual bound" -- the test file only
passes today because its invoker pre-shrinks the solve tolerance by 0.02x as a workaround
(`KrylovBattery.Invokers.fProxy.cs:232`, `SolveTol => TolValue * 0.02`), not because the solver
itself is honest.

### 3.4 `biCGStab` -- vulnerable, no verify anywhere

`r` is set once from `b - Ax` at init (`Krylov.BiCGStab.fProxy.cs:71-73`) and thereafter
propagated only via two in-place updates:
```
Krylov.BiCGStab.fProxy.cs:121   fProxy ss = Blas.axpyNormSq(-alpha, v, ref r);       // r := s = r - alpha*v
Krylov.BiCGStab.fProxy.cs:170   rr = Blas.axpyNormSq(-omega, t, ref r);              // r := s - omega*t
```
`x` is accumulated on a *separate* path (`x.addScaledInPlace(alpha, p/pHat)` at
lines 127-128/160/165, `x.addScaledInPlace(omega, r/sHat)` at 161/166). These two recurrences
(`r`'s and `x`'s) are only equal to `b - Ax` in exact arithmetic; nothing re-synchronizes them.
The DEVLOG's root-cause note (`TemplateSourceTests/DEVLOG.md:127-129`) names "biCGStab/idr's
pivot solves" (the `rho`/`rv`/`omega` denominators at lines 92, 113, 118, 151) as the same
near-zero-but-nonzero-pivot amplifier as minres/gmres. Convergence is declared at:
```
Krylov.BiCGStab.fProxy.cs:123-130   if (ss <= threshold) { x.addScaledInPlace(...); return Converged(... math.sqrt(ss)); }
Krylov.BiCGStab.fProxy.cs:172-173   if (rr <= threshold) return Converged(... math.sqrt(rr));
```
No fresh `b - Ax` is ever computed anywhere in this file (confirmed: "verify-at-exit" does not
appear in `Krylov.BiCGStab.fProxy.cs`).

### 3.5 `gmres` / `fgmres` -- vulnerable, and the most severe shape

`resnorm = math.abs(g[j+1])` (`Krylov.GMRES.fProxy.cs:142`, `Krylov.FGMRES.fProxy.cs:152`) is
the classic incrementally-Givens-rotated Hessenberg least-squares residual estimate -- and the
convergence decision happens **before `x` is even formed for this cycle**:
```
Krylov.GMRES.fProxy.cs:142-145
resnorm = math.abs(g[j + 1]);
total++;
k = j + 1;
if (resnorm <= thresh) { converged = true; break; }
...
// back-substitution + x accumulation happens AFTER this, at lines 148-176
```
The back-substitution that produces the actual solution update (`y[i] = sum / H[i,i]` at line
153) is exactly the second amplification site the DEVLOG names ("gmres/fgmres's
`y[i]=sum/H[i,i]`... a small-but-not-exactly-zero pivot... amplifies the update by orders of
magnitude") -- meaning the `converged=true` flag can already be latched *before* the step that
can blow `x` up even runs. `fgmres` (`Krylov.FGMRES.fProxy.cs:152-155`, back-substitution at
158-183) has the identical shape (it shares `gmres`'s Hessenberg/Givens machinery per its own
DEVLOG note). Both doc comments (`Krylov.GMRES.fProxy.cs:29`, `Krylov.FGMRES.fProxy.cs:31`)
already say "rnorm from the Arnoldi residual estimate" -- an explicit acknowledgment that this
value is not verified.

### 3.6 `idr` -- vulnerable, same shape as biCGStab

`R` is set once at init (`Krylov.IDR.fProxy.cs:84-87`) and thereafter propagated only via:
```
Krylov.IDR.fProxy.cs:164   rr = Blas.axpyNormSq(-beta, Gk, ref R);
Krylov.IDR.fProxy.cs:207   rr = Blas.axpyNormSq(-om, Q, ref R);
```
with `x` accumulated on a separate path (`x.addScaledInPlace(beta, Uk)` at line 165;
`x.addScaledInPlace(om, V/VHat)` at 208-209). Convergence declared at:
```
Krylov.IDR.fProxy.cs:168   if (rr <= threshold) { status = Converged; done = true; break; }
Krylov.IDR.fProxy.cs:212   if (rr <= threshold) { status = Converged; break; }
```
No verify anywhere in the file. The DEVLOG explicitly names "biCGStab/idr's pivot solves" (the
forward-substitution in `Msys` at lines 106-116/144-161, and the end-of-sweep `om` at line 199)
as the same near-zero-pivot amplifier class as minres/gmres.

## 4. Minimal fixes

All fixes reuse the exact `threshold`/`tol` variable already in scope at the call site -- no new
safety-factor constant is introduced, matching the precedent already set by `cg`'s and minres's
existing verify-at-exit blocks (neither adds slack beyond the solver's own threshold). Every fix
only fires **once, at the moment the tracked quantity first claims convergence** -- never per
iteration -- so the steady-state/common-case cost is unchanged and the only new cost on a
genuinely-converging run is exactly one extra `A.Apply` + `Blas.dot`, the same shape `cg`'s
existing fix already costs (see `KrylovVerifyAtExitTests.fProxy.cs`'s
`VerifyAtExitAddsExactlyOneApplyAndPreservesSolutionOnHealthySolve`). All fixes reuse Blas
kernels already used elsewhere in the same file, in the same call shapes, so determinism
(Strict FP, no new reassociation) is preserved by construction. **Breakdown exits are
untouched everywhere** -- that carve-out (Breakdown reports the unverified tracked value) is an
existing, intentional, library-wide convention (see `Krylov.PMinres` DEVLOG entry
2026-07-18, "Breakdown still reports the unverified phibar, matching every other solver's
Breakdown carve-out") and this task must not change it.

### 4.1 `minres` (`Krylov.MINRES.fProxy.cs:188-205`)

Delete the `M.IsIdentity` short-circuit at lines 190-192 and let both branches fall into the
verify block that already exists for the preconditioned case (lines 194-204) unchanged. That
block only touches `y` and `v` as scratch, both already idle at that point under identity too
(`z` -- the identity-unused buffer -- is never referenced), so no signature change and no new
buffer is needed:

```
if (phibar * phibar <= threshold)
{
    // Verify-at-exit (identity AND preconditioned): phibar can drift from the true ||b-Ax|| once
    // gamma has been floored (gammaFloor guard above) breaks the Givens rotation's unitarity.
    // y and v are both idle here. Fall through and keep iterating on a failed verify.
    A.Apply(in x, ref y);
    v.CopyFrom(in b);
    v.addScaledInPlace((fProxy)(-1), y);
    fProxy trueRR = Blas.dot(v, v);

    if (trueRR <= threshold)
        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(trueRR));
}
```
Do not touch the identity `MaxIterations` exit (line 212-213, still reports `phibar`) -- that
path already signals non-convergence honestly (`Solved == false`), so it is out of this fix's
scope (see §6).

### 4.2 `minresQLP` (`Krylov.MINRESQLP.fProxy.cs`, after line 377, before 379-382)

Gate only the two tolerance-based flags (`1` = `relres<=tol`, `2` = `relAresl<=tol`) against the
`finalRnorm` that is already computed unconditionally right above. Leave flags `-1`/`0` (exact
Lanczos-breakdown shortcuts), `3`/`4` (`chk1`/`chk2` stagnation -- effectively "relres has
underflowed to ~0", already trustworthy), and `5` (eigenvector-precision stagnation) untouched --
those are not the criterion the DEVLOG identifies as loose.

```
fProxy finalRnorm = math.sqrt(Blas.dot(r1, r1));

// flags 1/2 are minresQLP's own scale-normalized (rnorm/(Anorm*xnorm+beta1)) accept tests --
// looser than this library's ||b-Ax||/||b|| convention (documented ~13-14x looser even on
// well-conditioned systems; see the battery DEVLOG). Downgrade an unverified accept to an
// honest MaxIterations rather than report Converged with finalRnorm still above tol*||b||.
if ((flag == 1 || flag == 2) && finalRnorm > tol * math.sqrt(bb))
    flag = 8;

IterativeSolveStatus status;
if (flag == 8) status = IterativeSolveStatus.MaxIterations;
else if (flag == 6 || flag == 7 || flag == 9 || flag == -3) status = IterativeSolveStatus.Breakdown;
else status = IterativeSolveStatus.Converged;
```
`bb` (`Blas.dot(b,b)`, line 70) and `tol` (the method parameter) are both already in scope. Note
`iters` is left as whatever it was when flag 1/2 fired (not forced to `maxIter`) -- accurate
reporting of work actually done takes priority over the `IterativeSolveStatus.MaxIterations`
doc-comment's "ran the full budget" phrasing, which is prose, not an enforced invariant elsewhere
in this enum's usage.

**Deliberately out of scope for this minimal fix** (flag it as a follow-up, do not implement):
restructuring so a failed flag-1/2 verify resets `flag` to `flag0` and re-enters the loop
in-place (mirroring cg's continue-on-fail pattern exactly) instead of jumping straight to
`MaxIterations`. This file's flag-rollback logic (lines 360-366, specifically for flags
2/4/6/7) is delicate, reference-fidelity-ported code with extensive DEVLOG history -- an
in-loop retry is a larger, riskier change than this task's minimal-fix mandate and is better
done as its own reviewed increment.

### 4.3 `biCGStab` (`Krylov.BiCGStab.fProxy.cs`)

**Site (a) -- early exit, `Krylov.BiCGStab.fProxy.cs:123-130`.** `x` has NOT been committed at
this point, so the fix must verify a *trial* `x` before committing, to avoid double-applying
`alpha*p` if the verify fails and the code falls through to the standard path (which applies
`alpha*p` again at lines 160/165). `t` and `v` are both idle here (`t`: not yet written this
iteration; `v`: fully consumed by `alpha = rhoNew / rv` above, not read again until next
iteration's `A.Apply(in p, ref v)`) -- reuse them, no new buffers:

```
fProxy ss = Blas.axpyNormSq(-alpha, v, ref r);

if (ss <= threshold)
{
    // Verify-at-exit on a TRIAL x (not yet committed): t/v are both idle here. On a failed
    // verify, x is left untouched, so the standard stabilization step below applies alpha*p
    // exactly once (no double-apply).
    if (M.IsIdentity) { t.CopyFrom(in x); t.addScaledInPlace(alpha, p); }
    else              { t.CopyFrom(in x); t.addScaledInPlace(alpha, pHat); }
    A.Apply(in t, ref v);                     // v = A * (trial x)
    v.addScaledInPlace((fProxy)(-1), b);      // v = A*(trial x) - b; sign irrelevant, only dot(v,v) is used
    fProxy trialRR = Blas.dot(v, v);
    if (trialRR <= threshold)
    {
        x.CopyFrom(in t);
        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(trialRR));
    }
    // else: fall through to the standard stabilization step (t/v get overwritten below regardless).
}
```

**Site (b) -- main exit, `Krylov.BiCGStab.fProxy.cs:169-173`.** `x` is already fully committed
here (lines 158-167 ran before this point), so this is a direct copy of `cg`'s pattern. `v` is
idle (last read forming `rv` above, not touched again until next iteration's
`A.Apply(in p, ref v)`):

```
rr = Blas.axpyNormSq(-omega, t, ref r);
if (rr <= threshold)
{
    // Verify-at-exit (mirrors cg's). v is idle here. On a failed verify, r is left holding the
    // FRESH residual so the next iteration continues from a corrected state.
    A.Apply(in x, ref v);
    r.CopyFrom(in b);
    r.addScaledInPlace((fProxy)(-1), v);
    rr = Blas.dot(r, r);
    if (rr <= threshold)
        return MakeSolveInfo(IterativeSolveStatus.Converged, k + 1, math.sqrt(rr));
}
rho = rhoNew;
```

### 4.4 `gmres` / `fgmres`

The fix is a **one-line deletion per file** -- the true fresh-residual check already exists at
the top of every restart cycle (`Krylov.GMRES.fProxy.cs:74-80` / `Krylov.FGMRES.fProxy.cs:81-87`,
`A.Apply(x,w); v0=b-Ax; beta=sqrt(dot(v0,v0)); if (beta<=thresh) converged=true`) -- the bug is
that the inner-loop estimate is ALSO allowed to declare victory, pre-empting that fresh check.
Change:
```
Krylov.GMRES.fProxy.cs:145    if (resnorm <= thresh) { converged = true; break; }
Krylov.FGMRES.fProxy.cs:155   if (resnorm <= thresh) { converged = true; break; }
```
to:
```
if (resnorm <= thresh) break;
```
i.e. still exit the inner Arnoldi loop early (no point growing the basis further once the
estimate suggests convergence), still run the existing back-substitution + `x` update for this
cycle (unchanged, lines 148-176 / 158-184), but do not set `converged`. The outer
`while (total < maxIter && !converged)` then naturally loops back to the top, which reruns the
already-existing fresh `beta <= thresh` check against the just-updated `x` and is the ONLY
place that may now set `converged = true`. On a healthy/converging instance this adds exactly
one `A.Apply` + `Blas.dot` per solve and -- critically -- **does not change `total`/`iterations`**
(the outer while's top-of-loop re-check does not enter the inner `for` loop again once it
verifies), so no currently-passing exact-`iterations` assertion should need updating. If
`total == maxIter` is reached in the same step the inner estimate claimed convergence, the outer
loop exits without a chance to re-verify and correctly reports `MaxIterations` rather than a
possibly-false `Converged` -- the safe direction.

### 4.5 `idr`

**Site (a) -- in-sweep, `Krylov.IDR.fProxy.cs:164-172`.** `V`/`Q` are both idle from right after
they feed `Uk` (lines 119-140) until the next `k`-step's forward substitution reuses them --
reuse them here, no new buffers. Insert the verify inside the existing `if` (everything after it,
the `f[i] -= beta*Msys[i,k]` update, stays unconditional / unmoved):
```
rr = Blas.axpyNormSq(-beta, Gk, ref R);
x.addScaledInPlace(beta, Uk);
iter++;

if (rr <= threshold)
{
    // Verify-at-exit: V/Q are idle here. On a failed verify, R is left holding the fresh
    // residual (correct sign) so subsequent P[i]-dot-R work stays correct.
    A.Apply(in x, ref V);
    Q.CopyFrom(in b);
    Q.addScaledInPlace((fProxy)(-1), V);
    fProxy trueRR = Blas.dot(Q, Q);
    R.CopyFrom(in Q);
    rr = trueRR;
    if (trueRR <= threshold) { status = IterativeSolveStatus.Converged; done = true; break; }
}

if (k < s - 1)
    for (int i = k + 1; i < s; i++) f[i] -= beta * Msys[i, k];
```

**Site (b) -- end-of-sweep, `Krylov.IDR.fProxy.cs:207-212`.** `V`/`Q` become idle only after the
`x.addScaledInPlace(om, V/VHat)` update (both are read by it under the identity branch) -- insert
the verify after that, same buffer-reuse shape as site (a):
```
rr = Blas.axpyNormSq(-om, Q, ref R);
if (M.IsIdentity) x.addScaledInPlace(om, V);
else              x.addScaledInPlace(om, VHat);
iter++;

if (rr <= threshold)
{
    A.Apply(in x, ref V);
    Q.CopyFrom(in b);
    Q.addScaledInPlace((fProxy)(-1), V);
    fProxy trueRR = Blas.dot(Q, Q);
    R.CopyFrom(in Q);
    rr = trueRR;
    if (trueRR <= threshold) { status = IterativeSolveStatus.Converged; break; }
}
```
On a failed verify at either site, execution falls through with `done`/the outer `status`
untouched -- the `while (iter < maxIter && !done)` loop continues normally, using the
now-refreshed `R`.

## 5. Interaction with the `Forbids=IllConditioned` stopgap

`Assets/LinearAlgebra/CodeGen/TemplateSourceTests/fProxy/KrylovBattery.Invokers.fProxy.cs` gates
`minres`, `minresQLP`, `biCGStab`, `gmres`, `fgmres`, `idr` (plus `gcrodr`, `tfqmr`) off the
gallery's one `IllConditioned`-tagged `SymmetricIndefinite` matrix (Rosser) via
`Forbids: MatrixProfile.IllConditioned`, added on first battery-wiring per
`TemplateSourceTests/DEVLOG.md:123-141` because Rosser drove every one of them to a fresh
independent residual of `1e14`-`1e19` (`fProxyKrylovBatteryOracles.RelResidualDense`, computed
by the battery regardless of what status the solver reports).

**This fix does not, by itself, make the stopgap liftable**, for a specific reason worth being
explicit about: `KrylovSquareBatteryTests.fProxy.cs`'s check #1 already accepts *either*
`Converged` or `MaxIterations` as a passing status (`statusOk1 =
info1.status==Converged || info1.status==MaxIterations`, `KrylovSquareBatteryTests.fProxy.cs:89`)
-- the failure on Rosser is a **second, independent** assertion,
`relRes1 <= 10 * inv.Tol` (`KrylovSquareBatteryTests.fProxy.cs:92`), which is checked
unconditionally, regardless of status. This fix makes `status` HONEST (a solver will no longer
claim `Converged` unless a fresh residual actually backs it up) but does not make Rosser's
*numerical accuracy* any better -- the underlying near-zero-pivot amplification (§3) is a
separate, deeper problem (real breakdown-detection / regularization in each recurrence), already
called out as future work and explicitly out of scope for a "wiring task" in the same DEVLOG
entry. After this fix, minres/minresQLP/biCGStab/gmres/fgmres/idr should all report Rosser
honestly (`Converged` never lies) but most will likely still report `MaxIterations` (or
`Breakdown`) with a large residual -- which still fails check #1's `relRes1 <= 10*tol` assertion
as currently written, so **the `Forbids: IllConditioned` exclusion should stay** for these six
invokers under the battery's current check semantics.

What this fix DOES unlock, for whoever owns the battery next: check #1 could be tightened from
"the residual must always be small" to "IF `Solved` THEN the residual must be small" (i.e. only
assert the residual bound conditional on `status == Converged`, and separately accept an honest
`MaxIterations`/`Breakdown` with no residual bound at all) -- a strictly more correct thing to
assert once false-Converged is structurally impossible, and only once it's structurally
impossible. That is a test-infrastructure decision belonging to whoever next touches
`KrylovSquareBatteryTests.fProxy.cs` / `KrylovBattery.Invokers.fProxy.cs`, not this spec (this
task is solver-side only, per the brief) -- flagged here, not prescribed.

`minresQLP` is the one case where the current stopgap is arguably hiding a worse bug than "slow
convergence": before this fix it can report `Converged` with `finalRnorm` "~0.38" per the
invoker's own doc comment (`KrylovBattery.Invokers.fProxy.cs:210-212` /
`TemplateSourceTests/DEVLOG.md:142-149`) -- i.e. `Solved == true` while the residual is nowhere
close to small. After this fix that becomes an honest `MaxIterations`, closing the actual
correctness hole task #53 is about, even though it does not change whether Rosser passes the
battery's blanket residual check.

## 6. Explicitly out of scope for this fix

- Any change to `Assets/LinearAlgebra/Source/Generated/**` -- templates only, then regen.
- The near-zero-pivot amplification itself (real breakdown detection / regularization in the
  Givens/Hessenberg/shadow-space recurrences) -- a separate, larger effort; not attempted here.
- Loosening or removing any `Forbids: IllConditioned` entry in
  `KrylovBattery.Invokers.fProxy.cs` -- see §5.
- `minres`'s / `minresQLP`'s `Breakdown` exits, and `minres`'s identity-path `MaxIterations`
  exit (`Krylov.MINRES.fProxy.cs:212-213`, still reports unverified `phibar`) -- these already
  signal non-convergence honestly (`Solved == false`); only the `Converged` contract is broken
  today, so only `Converged` exits are touched.
- `minresQLP` flags `-1`/`0`/`3`/`4`/`5` and the in-loop retry alternative described in §4.2 --
  flagged as a follow-up, not implemented here.
- `gcrodr` and `tfqmr` -- plausibly already immune per their own doc comments (§2), but not
  audited to the same depth as the requested 7; a follow-up task should confirm rather than
  assume.
- Any change to `cg` -- already correct, included only as the reference pattern (§3.1).

## 7. Acceptance criteria

- [ ] `Krylov.MINRES.fProxy.cs`: the `M.IsIdentity` short-circuit at lines 190-192 is removed;
  both identity and preconditioned paths run the same verify block on a claimed `Converged`.
- [ ] `Krylov.MINRESQLP.fProxy.cs`: a flag-1/2 accept whose fresh `finalRnorm` exceeds
  `tol * sqrt(bb)` is downgraded to `MaxIterations` before `status` is computed; flags
  `-1,0,3,4,5` are unaffected.
- [ ] `Krylov.BiCGStab.fProxy.cs`: both Converged sites (early-exit `ss` and main `rr`) verify a
  fresh residual before returning `Converged`; the early-exit site never double-applies
  `alpha*p`/`alpha*pHat` to `x` on a failed verify.
- [ ] `Krylov.GMRES.fProxy.cs` and `Krylov.FGMRES.fProxy.cs`: the inner-loop `resnorm<=thresh`
  check no longer sets `converged = true` (still breaks the inner loop); only the top-of-cycle
  fresh `beta<=thresh` check may set it.
- [ ] `Krylov.IDR.fProxy.cs`: both Converged sites (in-sweep and end-of-sweep) verify a fresh
  residual before returning `Converged`, refreshing `R` (correctly signed) on a failed verify.
- [ ] For every one of the 6 fixed solvers: a "healthy solve" regression test (well-conditioned
  SPD/general dense matrix, no drift expected) shows **identical `x` and `iterations`** vs. the
  pre-fix behavior, and exactly one extra `A.Apply`-equivalent call on the converging path --
  mirroring `KrylovVerifyAtExitTests.fProxy.cs`'s existing
  `VerifyAtExitAddsExactlyOneApplyAndPreservesSolutionOnHealthySolve` for `cg`. (New test file or
  extension of `KrylovVerifyAtExitTests.fProxy.cs`/`KrylovBattery.Invokers.fProxy.cs` -- left to
  the test-writer agent.)
- [ ] A drift-firing regression per fixed solver: an instance where the unguarded/pre-fix
  recurrence would have claimed `Converged` with the true relative residual still above
  `tol` (Rosser via the battery gallery, or a constructed near-degenerate matrix, whichever the
  test-writer finds reproducible per-dtype) -- the guarded (post-fix) solver must never return
  `Solved == true` together with `||b-Ax||/||b|| > tol` (modulo the exact `threshold` convention each
  solver already uses). This is the core invariant task #53 is about; assert it directly,
  independent of whether Rosser specifically remains `Forbids`-excluded from the full battery.
- [ ] `KrylovSquareBatteryTests`/`KrylovBlockBatteryTests` and the full existing suite stay green
  with **no change to any currently-passing gallery matrix's reported `iterations`/`x`** (only
  previously-lying cases change behavior).
- [ ] `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/DEVLOG.md` gets one entry under
  `## Krylov.MINRES` / `## Krylov.MINRESQLP` / `## Krylov.biCGStab` / `## Krylov.gmres` /
  `## Krylov.idr` (per project convention -- DEVLOG, not code comments, carries the "why", the
  Rosser/gammaFloor root-cause narrative, and the task-#53 cross-reference).