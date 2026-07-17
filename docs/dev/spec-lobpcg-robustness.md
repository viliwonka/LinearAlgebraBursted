# Spec: LOBPCG spurious-zero-eigenvector collapse — diagnosis + fix

Status: researched + specced (Fable, web-grounded, 2026-07-17). Not yet implemented. Supersedes the
"minimum fix" hand-wave in [[lobpcg-tiny-penalty-collapse]].

## Diagnosis (named failure mode)

**Rank deficiency of the Rayleigh–Ritz basis producing spurious Ritz pairs, undetected because the
convergence test is not scale-invariant in ‖x‖.** Textbook instance of Duersch et al. 2018 §4.1 (their
worked example literally ends `θ₁ = 0 … spurious Ritz value` on an SPD matrix — same as our returned
`λ=0` with true `λ_min≈0.83`). NOT "ghost eigenvalues" (that's Lanczos), NOT "stagnation", NOT
"breakdown" (our `FactorGram` *passes* — part of the problem).

### Root defect (the actual bug)
The residual test (`LOBPCG.fProxy.cs:175-202`) is `‖A x_i − λ_i B x_i‖ ≤ tol·max(|λ_i|,1)`. `r(x)` is
**linear in x**, RHS is independent of ‖x‖, so any shrunken iterate passes and **x=0 passes exactly**
(λ≈0, r≈0). The test silently assumes ‖x‖_B=1 but nothing maintains that after `UpdateActiveBlock`
(the new rows `X_new = S·C` are B-unit only to the accuracy of the Gram Cholesky L, which is garbage in
float on the penalty pencil). Downward norm drift self-amplifies (smaller row → smaller residual → locks
at `lockTol=0.1·tol`), and the (d1) re-deflation block (`:258-276`) guards renormalization with
`if (bn2>0)`, so an annihilated row stays an **exact zero row forever** = a self-certifying fixed point.

### Why only SMALL penalty-conditioned systems
1. **Block-to-dimension ratio.** Demo runs k=4 + guard=4 → kWork=8; 1×1×1 frame n=24 → RR basis
   m=3·kWork=24=n: [X,W,P] tiles the whole space → guaranteed near-linear-dependence every iteration.
   Larger frames push 3·kWork≪n and it disappears. **SciPy refuses to run LOBPCG when 5k>n** (falls back
   to dense eigh); our repro is 5·8=40>24 — SciPy would never have run it.
2. **Penalty conditioning in float.** penalty 1e3 on pinned DOFs → cond(K)~10²–10³; Gram squares it
   (cond(VᵀBV)=cond(V)²); `FactorGram`'s gate `MinMaxDiagRatio ≥ sqrt(eps)≈3.45e-4` accepts cond(L)~3000
   → Gram cond ~1e7 → in float (eps~1.2e-7) zero trustworthy digits. The Ritz-envelope safeguard can't
   catch λ≈0 either (envelope ≈ ±3e6 for this pencil).

**Premise correction vs the memory note:** the residual is NOT stale (AX/BX are fresh matvecs at the exit,
`:359-360`); the certificate is invalid purely because ‖x‖ is absent from the test.

## Convergence criteria across reference impls
- **Duersch Eq. 9 (backward-stable):** `‖r_i‖ / ((‖A‖ + |θ_i|·‖B‖)·‖x_i‖) ≤ τ`. ‖x‖ explicit; norm
  estimates may be Frobenius-sketch LOWER bounds (errs strict, never lax).
- **SciPy:** absolute `‖r‖ ≤ tol`, but B-orthonormalizes X EVERY iteration so ‖x‖_B≡1 structurally;
  Cholesky-fail → warn + return best-residual iterate (non-converged); restart on 2²⁰× residual jump;
  dense fallback for 5k>n.
- **Anasazi/BLOPEX/MATLAB:** |θ|-relative with explicit ‖x‖; explicit `failureFlag`/`LOBPCGRitzFailure`.
- **Ours:** no ‖x‖ anywhere → the one thing they all avoid.

## Minimum fix (honest status, no algorithm change) — all Burst-safe/Strict-deterministic/zero-alloc
1. In the residual loop accumulate `xb2 += X[i,c]*BX[i,c]` → `xBnorm = sqrt(max(xb2,0))`.
2. Convergence/lock test → Duersch Eq. 9 shape: `rnorm ≤ tol·(normAEst + |λ_i|·normBEst)·‖x_i‖₂`.
3. Norm estimates, zero extra matvecs: right after the entry ApplyBlock on the Euclidean-orthonormalized
   seed, `normAEst = ‖AX‖_F/sqrt(kWork)`, `normBEst = ‖BX‖_F/sqrt(kWork)` (=1 on B=I). Frobenius-sketch
   lower bounds (Duersch). Optionally tighten `normAEst = max(normAEst, |λ_i|)` in fixed order.
4. **Degenerate detection → new status.** If `xBnorm < normFloor` (spec 0.25) mark the pair degenerate:
   never locks, never counts toward converged; if it persists among the k wanted at exit, return
   `IterativeSolveStatus.Degenerate` (new enum value, additive; contract: "numerically degenerate iterate
   / collapsed RR basis; returned pairs NOT certified — treat as non-converged"). `LOBPCGInfo.Solved`
   stays `status==Converged`, so every `if (lobpcg(...))` call site becomes honest automatically.
5. Fix the zero-row fixed point in (d1) `:265`: replace `if (bn2>0)` with a real threshold; below it,
   **reseed the row deterministically** (fixed seed keyed by (iter,i)) + Deflate + B-normalize, instead
   of leaving an annihilated row.
6. Block-size sanity: validate/document `3·kWork ≤ n` in the arena guard overloads; for `3·kWork > n`
   don't throw (existing callers) but never report Converged without the exit certificate (item 4 covers).

Acceptance: 1×1×1 frame repro returns `status != Converged` (Degenerate), `converged==0`, no returned
row with ‖x‖_B<0.25 flagged converged; all 6317 tests stay green (smoke tests assert values/orthonormality
not iteration counts).

## Robustness fix (makes the tiny penalty case actually solve) — each independently shippable
1. **Restore the B-unit invariant on X every iteration** (cheapest 80%): after UpdateActiveBlock + the
   fresh ApplyBlock (`:349-369`), B-normalize each active row (`inv=1/sqrt(dot(X_i,BX_i))`, scale
   X_i/AX_i/BX_i by linearity — no matvec; reseed if the dot is below threshold). Kills the self-scaling
   false certificate AND keeps the next Gram diagonal ≈1 (Duersch: diagonal scaling dramatically lowers
   cond(R) for free).
2. **Upgrade `OrthonormalizeBlockB` from Cholesky-QR+ridge to SVQB-with-dropping** (Duersch Alg. 6): scale
   Gram by D=diag(G)^{-1/2}, eig-decompose DGD=ZΘZᵀ via `Eigen.symmetricInPlace` (m≤24, Temp), keep
   J={j:θ_j>θ₁·τ_drop} with `τ_drop = m·eps·c` (c≈10), output `V·D·Z(:,J)·Θ(J,J)^{-1/2}` (AV/BV by the
   same combination). Drop the collapsed W/P direction instead of ridge-inflating it into noise. The RR
   machinery then needs per-block widths (nx,nw,np) — `BuildProjected` already takes counts, so it
   generalizes; the one real structural change.
3. **Tighten the combined-Gram gate to the cube rule** (Duersch §5.3): `MinMaxDiagRatio(L) ≥ (eps·c)^{1/3}`
   (float ≈ 0.005–0.01, cond(L)≲100–200), since L⁻¹ is applied 3× in RR. Failure routes to drop-P →
   stall → honest Degenerate/MaxIterations.
4. **Do NOT** fix with a bigger ridge (converts rank-deficiency into plausible noise), restarts (re-enters
   the same Gram), or a better preconditioner (not the defect).

## Penalty-conditioning verdict (game/structural context)
Penalty isn't fundamentally hostile to LOBPCG — it's hostile to FLOAT LOBPCG via cond(K)=O(penalty/stiff)
→ cond(Gram)=O(that²). In order: (1) prefer reduction/elimination BCs (delete pinned DOFs from the BSR —
exact, shrinks n, removes the artificial spectrum top; trivial index filter at assembly) — right default
for demos/README; (2) if penalty in float, keep it within ~3 decades of physical stiffness; (3)
shift-and-invert NOT recommended (needs a solve per matvec, defeats matrix-free BSR).

## Test vectors to add
Assert per case: `Solved ⇒ (λ matches dense Eigen.symmetricInPlace to tol AND min‖x_i‖_B ≥ 0.5)`;
`!Solved` always acceptable — NEVER require convergence on hard cases, only forbid false certificates.
1. **Tiny penalized frame** (the repro, minimal = unit-cube truss, n=24, EA=8, penalty∈{1e3,1e5}). Add a
   gallery generator `fProxyPenalizedGrid3D(ref Arena, nx,ny,nz, EA, penalty)` (dense+BSR) so tests don't
   depend on demo code. Run both the pathological (k=4,guard=4 → 3·kWork=n) and sane (guard=0) configs.
2. Penalized 1-D Laplacian: `EA·Laplacian1D(16) + penalty·(e₁e₁ᵀ+eₙeₙᵀ)`, penalty=1e4.
3. Wilkinson W₂₁⁺ (have) + **add `fProxyWilkinsonMinus`** — near-double pairs.
4. Clustered-bottom Laplacian near-multiple variant (gridX=gridY+1).
5. Läuchli LᵀL, eps=sqrt(float eps): assert λ≥0, honest status.
6. Rank-deficient B = diag(1,…,1,0): must return non-Converged (contract test for the new status).
7. Adversarial warm starts: exact eigenvectors (0-iter certify), and a zero-row + k−1 exact vectors.
8. Through-IJob variants of 1 & 7 (per [[lobpcg-burst-eigenvector-bug]] lesson).

## Sources
Duersch/Shao/Yang/Gu 2018 (arXiv:1704.07458, SIAM J.Sci.Comput. 40(5)) — Eq.7/8/9, §3 ill-conditioning,
§4.1 spurious θ₁=0 + svqbDrop, §5.3 cond(R)³ rule. Hetmaniuk&Lehoucq 2006 (JCP 218). Stathopoulos&Wu 2002
(SVQB). Knyazev 2001 (original). SciPy lobpcg docs+source. Anasazi LOBPCGSolMgr. BLOPEX/Octave lobpcg.
Wikipedia LOBPCG ("instability prominent in single precision"). de Prenter et al. arXiv:2208.08538
(penalty O(1/ε) conditioning).
