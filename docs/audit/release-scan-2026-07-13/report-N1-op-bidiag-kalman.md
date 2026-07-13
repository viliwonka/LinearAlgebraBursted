# Narrow scan N1 — TemplateSource/OP (Bidiag.fProxy.cs .. Kalman.UKFCache.fProxy.cs)

Partition: 24 files, case-insensitive alphabetical from `Bidiag.fProxy.cs` through
`Kalman.UKFCache.fProxy.cs` inclusive. Every line of every file read.

Files: Bidiag.fProxy, Bidiag.Workspace.fProxy, Blas.ColumnScaling.fProxy, Blas.Fused.fProxy,
Blas.Triangular.fProxy, BoolOP, CHO.fProxy, CHOP.fProxy, CHOP.Workspace.fProxy, Control.fProxy,
Control.Info, Easing.fProxy, Eigen.fProxy, Eigen.Info, Eigen.LanczosWorkspace.fProxy,
Eigen.SymWorkspace.fProxy, FFT.fProxy, FFT.Workspace.fProxy, GenOP.fProxy, Kalman.fProxy,
Kalman.Info, Kalman.State.fProxy, Kalman.UKF.fProxy, Kalman.UKFCache.fProxy.

All 24 files are fProxy (float/double) templates or `//singularFile//` enum/info files. There are
NO iProxy/bool-numeric templates in this partition, so integer overflow / division-truncation /
unsigned-underflow (W4 int concerns) do not apply here. Per-type constants everywhere route through
`Consts.fProxy*` tokens (fProxyEpsilon, fProxyZeroThreshold, fProxySqrtEps, fProxyCholBlockMinN,
fProxyCholPivotBlockMinN), which substitution rewrites to the correct float*/double* member — so the
epsilon/threshold/block-gate constants are correctly precision-split. Confirmed clean on W4.

## Findings

### 1. HIGH — `Eigen.valuesQR` mutates/destroys A but lacks the `InPlace` suffix
File: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Eigen.fProxy.cs:1606` (and default overload :1892).

`public static unsafe EigenInfo valuesQR(ref fProxyMxN A, ref fProxyN eigenvaluesReal, ref fProxyN eigenvaluesImag, int maxIterPerRoot)`
takes `A` by `ref` and its XML doc states plainly "A is DESTROYED (overwritten during
reduction/iteration)". The naming canon (docs/naming-style-guide.md; confirmed as a wide-pass HIGH
and addendum pattern #3) requires the `InPlace` suffix exactly when a method destroys/overwrites its
input. Every sibling destructive eigensolver in the SAME file honors this: `decompInPlace`,
`valuesSymmetricInPlace`, `symmetricInPlace` all take `ref` + destroy `A` + carry `InPlace`.
`valuesQR` is the lone destructive solver missing it — an API-surface inconsistency that becomes a
breaking rename after v1.0.
Concrete failure: a caller who assumes non-destructive semantics from the un-suffixed name (matching
e.g. `Bidiag.values`, which explicitly does NOT modify A) silently loses A.
Fix direction: rename to `valuesQRInPlace` (with the default-arg overload); update the cross-refs
that name it — `FFT.fProxy.cs:11` ("Like Eigen.valuesQR…") and `Eigen.Info.cs:81`. Pre-release is the
cheap window for this breaking rename.

### 2. LOW — `fProxyUKFCache` doc-comment duplicates measured benchmark evidence already in DEVLOG
File: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.UKFCache.fProxy.cs:123-133`.

The default-ctor XML doc carries measured float32 numbers and a rejected-alternative verdict:
"measured (float32 prototype) to produce catastrophic cancellation … a 1e-3 tracking error … blew up
to ~1 with alpha=1e-3, vs ~1e-6 with alpha=1". The comment policy (CLAUDE.md) puts benchmark numbers,
perf verdicts and rejected alternatives in DEVLOG only — and `OP/DEVLOG.md:375-390` (## Kalman.UKF)
ALREADY records this exact evidence with fuller numbers. This is duplicated dev-history in shipped
source.
Fix direction: cut the measured-evidence clause down to the contract ("defaults alpha=1/beta=2/kappa=0;
alpha=1 keeps Wc[0] non-negative for kappa=0"); the "why 1e-3 was rejected" narrative stays in DEVLOG
(add `(was Kalman.UKFCache.fProxy.cs:124)`).

### 3. LOW — Easing class XML doc uses a float-suffixed literal in a double-generating template
File: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Easing.fProxy.cs:9`.

The `<summary>` example `new fProxyEasing.SmoothStep().Eval(0.3f)` substitutes verbatim into the
double variant as `new doubleEasing.SmoothStep().Eval(0.3f)` — the `f` suffix is wrong for the double
build (harmless as prose, but a copy-into-IDE example reads oddly / narrows). Addendum pattern #6, but
doc-only (not code), hence LOW.
Fix direction: drop the suffix (`Eval((fProxy)0.3)` is awkward in prose; simplest is `Eval(0.3)`), or
reword to avoid a typed literal.

### 4. LOW — `BoolOP` carries a class-level `[BurstCompile]` no sibling OP class has, placed before its doc comment
File: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/BoolOP.cs:8-14`.

`[BurstCompile]` sits on line 8, ABOVE the `<summary>` doc (lines 9-13), above the class (line 14) —
unusual ordering (attributes normally follow the doc comment). More notably, `boolComp` is the only
op-class in this partition (vs Blas/CHO/CHOP/Eigen/FFT/Generate/Kalman, none of which are
`[BurstCompile]`-annotated) carrying a class-level attribute; on a plain static extension-method class
with no `[BurstCompile]`-marked members it is effectively a no-op. Cosmetic/consistency only.
Fix direction: remove the vestigial class attribute (or, if intentional, move it below the doc comment
and document why boolComp alone needs it).

### 5. LOW — internal "(spec estimate)" reference in a code comment
File: `Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Control.Info.cs:141`.

`SDA_MAX_ITER` comment says "…reach machine-precision-class residuals in ~10-25 steps (spec estimate);
50 is a generous margin". "(spec estimate)" is a soft internal-spec reference; the surrounding rationale
is contract-adjacent but the parenthetical is DEVLOG material. (The Control.fProxy.cs header's
"port of Chiang-Fan-Lin Algorithm 2.1" and Kalman's "FilterPy (rlabbe/filterpy, MIT)" are literature/
attribution references and are legitimate — flagged here only to note they were reviewed and left.)
Fix direction: drop "(spec estimate)".

## Areas confirmed clean (checked, no defect)

- **Bidiag** (fProxy + Workspace): Householder col/row reflectors, left/right applies, U backward
  reconstruction, values-only path (NR e[0]=0 convention), dim guards, workspace sizing, Temp
  alloc/dispose pairing, validate-before-alloc ordering — all correct. A is genuinely not modified
  (works on ws.W copy), matching its doc.
- **Blas.ColumnScaling / Blas.Fused / Blas.Triangular**: NaN-safe `buildJacobiScale` (c>0 guard),
  fused Krylov wrappers forward operand roles correctly, all triangular/TRSM/transposed compact-LU
  solves index row-major correctly with consistent square/tall + size guards; pivot indirection
  (RP[r]) applied consistently; unguarded-diagonal precondition documented uniformly.
- **CHO / CHOP (+ Workspace)**: right-looking + blocked POTRF/PSTRF, `!(d>0)` NaN-safe pivot rejection
  before sqrt, self-alias read-before-write ordering for the InPlace paths, rank-deficient min-norm
  Gram path with Tikhonov-ridge retry, all Temp buffers disposed on every return branch (including the
  early Indefinite/NotPD exits). Multi-RHS mirrors vector forms.
- **Control (+ Info)**: SDA doubling, shared RiccatiStep kernel, warm/schedule/lqg entry points; every
  scratch Temp disposed; double-precision scalar locals never stored back into fProxy buffers;
  last-known-good-iterate convention honored; blowup/divergence guards consistent. `LQRStatus`/
  `LQRInfo`/`LQGInfo` diag structs well-formed.
- **Easing / GenOP**: struct-functor curves and generators; endpoint-pinning in linspace/sample;
  sigma>0 and N>=1 guards; validate-before-alloc in gaussianKernel2D. math.* calls resolve for both
  float and double.
- **Eigen (+ Info / LanczosWorkspace / SymWorkspace)**: power / inverse-power iteration (distinct-buffer
  alias guards, deterministic zero-seed, div-by-zero guards on norms), Lanczos with twice
  reorthogonalization + early-breakdown padding (Gershgorin-scaled decoupling), Householder+QL
  values/vectors (denormal-underflow deflation guard, [NoAlias] jacobiRotate/francisRow on distinct
  rows), Hessenberg+Francis QR. Only defect = the valuesQR naming (Finding 1).
- **FFT (+ Workspace)**: radix-2 / radix-4 / mixed-radix cores, conjugate-trick inverse, block-recurrence
  twiddle re-seeding, rfft/irfft two-for-one packing, alias guards on outputs, `(long)k*t % n` overflow
  guard in the DFT twiddle. Workspace sizing guards match the factory.
- **Kalman (+ Info / State / UKF / UKFCache)**: Joseph-form update via CHOP on the transposed system
  (no explicit inverse), LQR/KF DARE duality reuse of SDACore with Q/R data-norm rescale-then-unscale
  (Kss scale-invariant), EKF numeric-Jacobian helpers, UKF Van der Merwe sigma points via pivoted
  Cholesky with degenerate-spread fallback, x/P left UNCHANGED on Indefinite innovation covariance.
  InPlace/addScaled/mul/sub wrappers all pass operand roles correctly; Temp alloc/dispose balanced on
  every branch.

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 1     |
| MEDIUM   | 0     |
| LOW      | 4     |

HIGH: `Eigen.valuesQR` (Eigen.fProxy.cs:1606) — destructive `ref A` eigensolver missing the `InPlace`
suffix that all its destructive siblings carry; breaking rename is cheap only pre-release.
