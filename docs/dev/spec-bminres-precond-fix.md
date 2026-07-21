# Spec: fix for Krylov.bminres preconditioned path (#49)

Status: root cause DIAGNOSED + fix VERIFIED in the numpy reference (2026-07-22). Not yet ported.

## The bug

`BlockNormalizePrecond` itself (the M⁻¹-weighted Cholesky normalization `G = W M⁻¹ Wᵀ`,
`V = Beta⁻¹ Z`, `Z = M⁻¹ W`) is CORRECT -- it produces genuinely M-orthonormal V
(`V M Vᵀ = I`, verified). The bug is in the block-Lanczos RECURRENCE that feeds it
(`Krylov.Block.MINRES.fProxy.cs` main loop, lines ~316-335, the `!M.IsIdentity` path).

The recurrence subtracts the α/β Lanczos terms in **V-space** (the M-orthonormal vectors
`Vcur`/`Vprev`) and then applies `M⁻¹` to the whole result inside the normalizer:

```
Wk = A·Vcur
if k>=1: Wk -= Betaᵀ·Vprev      // V-space
Alfa = BlockGram(Vcur, Wk)      // (after beta subtraction)
Wk -= Alfa·Vcur                 // V-space
// normalize: Z = M⁻¹ Wk, G = Wk M⁻¹ Wkᵀ
```

For M=I this is correct (V-space == r-space up to β). For M≠I the `M⁻¹` hits the
subtraction terms too, so the resulting Lanczos vectors are NOT M-orthogonal and the
tridiagonalization is wrong. This is why the scalar (`Krylov.MINRES.fProxy.cs`)
converges but the block does not: the scalar keeps the UNPRECONDITIONED residuals
`r1,r2` and subtracts `(β/oldβ)·r1`, `(α/β)·r2` in r-space (`z = M⁻¹r`, `β = √⟨r,z⟩`).

## The fix (verified)

Maintain the UNPRECONDITIONED residual blocks (r-space), mirroring the scalar. The correct
recurrence (only M⁻¹ needed) is:

```
Wnext = Vcur·A  −  Alfa·(Beta⁻¹ Wcur)  −  Betaᵀ·(Beta_prev⁻¹ Wprev)
```

where `Wcur`/`Wprev` are the unpreconditioned residual blocks (Wcur = the W that was
normalized to produce the current `Vcur`/`Beta`; W0 = R0 = B − A X0), and `Beta`/`Beta_prev`
are their s×s Cholesky normalization factors. `Alfa = BlockGram(Vcur, Vcur·A)` computed from
`Vcur·A` BEFORE any subtraction (= Vcur A Vcurᵀ). `Beta⁻¹ Wcur` and `Beta_prev⁻¹ Wprev` are
lower-triangular solves. Then normalize `Wnext` (BlockNormalizePrecond) → BetaNext, Vnext,
and roll: `Wprev←Wcur`, `Wcur←Wnext`, `Beta_prev←Beta`, `Beta←BetaNext`.

Derivation: from formulation-A (M-orthonormal V-space, `D = Vcur·A·M⁻¹ − Alfa·Vcur −
Betaᵀ·Vprev`, normalize `G = D M Dᵀ`), which needs M-forward; but `Vk·M = Beta_k⁻¹ Wk`, so
`W = D·M` gives the M⁻¹-only r-space form above. Reduces EXACTLY to the scalar at s=1.

## Verification (reference/wip-bminres/, gitignored)

- `precond_gt2.py`: formulation-A M-orthonormal Lanczos, `V M Vᵀ = I` to 2.5e-13.
- r-space recurrence exact at s=1 (5e-17), and formulation-A ≡ r-space at s>1 (both agree).
- DECISIVE (`precond_minimal.py` + comparison): on 58 cases where the CORRECT recurrence
  converges <1e-8, the C#'s exact WRONG recurrence diverges on ALL 58.
- Hard-indefinite cases that fail in the numpy harness fail at M=I TOO (shared with the
  identity path, since formulation-A(M=I) is byte-identical to the verified `bminres_ref`) --
  NOT a preconditioning issue; the real C# uses rank-revealing CHOP to deflate those.

## Port notes

- Branch the Lanczos step on `M.IsIdentity`: keep the current V-space recurrence for identity
  (correct + verified 522/522), use the r-space recurrence for precond.
- New scratch: `Wcur`, `Wprev` (s×n unprec residual blocks), `Beta_prev` (s×s), plus a buffer
  for the tri-solve results (or reuse `T`). `Wcur` must be SAVED before `BlockNormalizePrecond`
  overwrites `Wk` with `Vout` (the precond normalize aliases Vout into Wk).
- Un-gate: remove the `NotSupportedException` at line ~209-210; the battery's check #5 (real
  BlockJacobi M) then exercises it. Verify the whole block suite stays green.
- The MINRES machinery (Omega/Gamma/Phibar/W/X) is UNCHANGED between identity and precond.
