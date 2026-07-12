# Release scan 2026-07-12 — area: kalman-family (post-scan code)

{"total":1,"confirmed":1,"uncertain":0,"unverified":0,"refuted":0,"high":0,"medium":0,"low":1}

## Scope

- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.State.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.Info.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.UKF.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.UKFCache.fProxy.cs
- Assets/LinearAlgebra/CodeGen/TemplateSource/Interfaces/KalmanModel.fProxy.cs

## Findings

### 1. [low/performance/CONFIRMED] Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.fProxy.cs:73 — UpdateCore recomputes H·P with a fresh GEMM though PHt already holds its transpose

**Evidence**

```
Line 58-59 compute `PHt = P Hᵀ` (n x m) to form Smeas. Line 73 then does
`Blas.dot(in H, in s.P, ref Xt)` to build Xt = H·P (m x n). Because P is
symmetric, Xt == PHtᵀ exactly, so this is a second O(m·n²) matrix-matrix
product where an O(m·n) transpose of the already-computed PHt would suffice.
PHt is otherwise kept alive but unused after Smeas (disposed at line 110).
```

**Verifier**

In Kalman.fProxy.cs:73, UpdateCore computes Xt = H * s.P via a full GEMM (cost O(m*n^2)) even though PHt = P * H^T (n x m) was already computed at line 59 and remains live until disposal at line 110. Since s.P is a documented Kalman symmetric invariant AND is provably symmetric at UpdateCore entry (every write path — PredictCovarianceCore line 41 and UpdateCore's own tail line 98 — passes through Control.SymmetrizeInPlace before the CopyFrom into s.P), we have H*P = H*P^T = (P*H^T)^T = PHt^T exactly in math. Replacing the GEMM at line 73 with Blas.trans(in PHt, ref Xt) is dimensionally valid (Blas.trans in OP.Dot.fProxy.cs:229 requires T.M_Rows == A.N_Cols and T.N_Cols == A.M_Rows; PHt is n x m and Xt is m x n — matches), non-aliasing (separate Allocator.Temp allocations), and reduces cost from O(m*n^2) to O(m*n). The DEVLOG's Kalman section (lines 248-317) documents the transposed-CHOP-solve rationale and the SDA scaling bug but says nothing that defends the redundant H*P recomputation as an intentional design choice — no by-design defense on record. Severity low is correct: this is a performance win only, correctness unchanged since P-symmetry is architecturally enforced.

**Suggested fix**

Replace the `Blas.dot(in H, in s.P, ref Xt)` at line 73 with `Blas.trans(in PHt, ref Xt)` (PHt is n x m, Xt is m x n); this reuses the existing GEMM result instead of doing a redundant one, correctness unchanged since P is symmetric.

## Scanner notes

Scanned all 6 target files in full plus CHOP.fProxy.cs, CHOP.Workspace.fProxy.cs, and Control.fProxy.cs for cross-verification. The special-lens hazards called out in the brief were all checked and found correct: (1) SDA duality mapping SDACore(Aᵀ, Hhat, Q, R) matches the filter DARE algebraically against Control's LQR DARE S=Q+AᵀSA-AᵀSB(R+BᵀSB)⁻¹BᵀSA; (2) Q/R joint rescale scales Sigma by the same c (verified from the DARE directly), Sigma is unscaled via invScale=1/scale before the gain extraction which uses the ORIGINAL R, and the zero guard math.max(dataNorm, fProxyZeroThreshold) prevents div-by-zero; (3) Joseph form (I-KH)P(I-KH)ᵀ+KRKᵀ with explicit SymmetrizeInPlace is correct including term2 K R Kᵀ via Xt==Kᵀ; (4) transposed CHOP gain solve S·Kᵀ=HP yields Kᵀ=S⁻¹HP and K=PHᵀS⁻¹ correctly (S,P symmetric); (5) UKF sigma scatter diff[Piv[i]]=L[i,k]·scale matches CHOP's PᵀΣP=LLᵀ convention (M=ΠL, MMᵀ=Σ) — this is the flagged hazard and it is handled correctly, and the DEVLOG documents catching an earlier wrong version; (6) negative-W0 recombination is symmetrized; (7) no scratch aliasing across predict/update in fProxyKFState buffers (covariance uses old P, x updated last); (8) Temp disposal on both InnovationSolveFailed (UpdateCore) and ukfUpdate Indefinite paths is leak-free (the if-block-only allocations K/Ky/PxzK etc. are never reached on failure); (9) MMax contract enforced in updateFixed (H MMax x n, z len MMax, Kss n x MMax, yFast sized mMax). No numerical, logical, pointer, or naming defects found. Files are effectively clean.
