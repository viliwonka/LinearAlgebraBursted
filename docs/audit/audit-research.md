# LinearAlgebraBursted — Algorithm Audit Report

*Historical document — method names predate the 2026-07 solver-API rework (see
docs/spec-solver-api-rework.md for the mapping).*

**Date:** 2026-06-28  
**Auditor:** Claude Sonnet 4.6 (automated cross-reference audit)  
**Scope:** All algorithm implementations in `Assets/LinearAlgebra/CodeGen/TemplateSource/`  
**Method:** Source reading + web research against LAPACK, EISPACK, and primary literature  

---

## Executive Summary

The library is in good algorithmic shape. All twelve core numerical algorithms (LU, Cholesky, SVD, Jacobi eigenvalue, Francis QR, CG, FFT, Box-Muller, Householder QR, QRCP, k-means++, Catmull-Rom) are correctly implemented against their canonical references with no critical or high-severity defects. All DSP window function coefficients are exact. The random-distribution ICDFs are textbook-correct.

One medium-severity issue was found: the Gallery `DingDong` matrix uses a denominator formula shifted by +1 relative to both MATLAB's `gallery('dingdong')` and Higham's test-matrix toolbox, producing a different matrix than the canonical reference while preserving qualitative properties. One low-severity issue was found in the Pivoted Cholesky tolerance scan (all-entries vs diagonal-only, harmless for PSD inputs).

The task prompt's cross-check item for the Rosser eigenvalues is incorrect — the library's comment is right.

---

## Algorithm Subsections

### 1. LU Decomposition with Partial Pivoting

**File:** `OP/LU.fProxy.cs`  
**Algorithm:** Gaussian elimination with partial pivoting (column-maximum strategy)  
**References:**
- Golub & Van Loan, *Matrix Computations* 4th ed., §3.4 (LUP factorization)
- LAPACK User's Guide, `DGETRF`

**Verdict: MATCHES**

Three variants are present and all correct:
- `luDecompositionNoPivot`: Separate L/U matrices, explicit zero below U diagonal. Correctly aborts on zero pivot before dividing.
- `luDecomposition`: Separate L/U with partial pivot. Searches rows k..m-1 for the max absolute value, swaps rows in both U and the already-computed leading columns of L. Correct.
- `luDecompositionInplace`: Compact packed form using a `Pivot` array (LAPACK style). Stores multipliers below diagonal in-place. The diagonal scan `LU[P[k], k]` correctly reads the pivoted row. Correct.

`LUSolve` applies the inverse permutation to b, then solves Ly=b (forward, unit diagonal) and Ux=y (back substitution). `determinant` correctly computes `P.Sign * Π LU[P[i],i]`.

**Notes:**
- `SolveLowerTriangularLU` has no division by the diagonal (line 76: `x[r] = (x[r] - sum)`). This is intentional and correct: the compact LU form stores unit multipliers in L with diagonal = 1 implicitly omitted.

---

### 2. Cholesky Decomposition

**File:** `OP/Cholesky.fProxy.cs`  
**Algorithm:** Column-by-column (outer-product) Cholesky; pivoted variant follows LAPACK xPSTRF  
**References:**
- Golub & Van Loan, §4.2.1
- LAPACK `DPSTRF` source (netlib.org/lapack)
- Higham, *Accuracy and Stability of Numerical Algorithms*, §10.3

**Verdict: MATCHES**

`choleskyDecomposition`: Standard outer-product form. Computes L[j,j] = sqrt(A[j,j] - Σ L[j,k]²), then off-diagonal L[i,j] = (A[i,j] - Σ L[i,k]L[j,k]) / L[j,j]. The guard `!(diag > 0)` is NaN-safe and fires before the sqrt. In-place aliasing (A=L) is documented as safe. Correct.

`choleskyDecompositionPivot`: Full LAPACK xPSTRF algorithm: symmetric permutation (largest remaining diagonal first), pivot column factored, trailing Schur complement updated symmetrically. Stops when max remaining diagonal ≤ stopTol = n·ε·absScale. Returns false if min diagonal goes below −stopTol (indefinite detection). The extra check for significant off-diagonal entries when the diagonal falls below stopTol prevents silently accepting an indefinite matrix as rank-0 PSD. Correct and more defensive than LAPACK.

`choleskyPivotSolve` (rank-deficient path): Constructs the Gram matrix G = L₁ᵀL₁ (rank × rank), applies Cholesky twice to compute G⁻²g, then recovers x = L₁z. The Tikhonov ridge on the inner G factorization is a correct numerical safeguard. Correct.

`SolveUpperTriangularTransposed`: Back-solves Lᵀx=b by reading L[c,r] instead of materializing Lᵀ. Correct.

**Notes:**
- Tolerance scan: `absScale` is computed as max over ALL entries of W, while LAPACK `dpstrf` scans only the initial diagonal. For genuine PSD input the largest diagonal dominates, so the results are identical. For non-PSD input the library's scan is slightly more conservative — harmless.

---

### 3. SVD — One-Sided Jacobi

**File:** `OP/SVD.fProxy.cs`  
**Algorithm:** One-sided Jacobi (Brent-Luk / Drmac-Veselic cyclic sweep)  
**References:**
- Drmac & Veselic, "New Fast and Accurate Jacobi SVD Algorithm," *SIAM J. Matrix Anal. Appl.* 2008
- Golub & Van Loan §8.6.2

**Verdict: MATCHES**

Per-pair (p,q) inner products: α = ‖u_p‖², β = ‖u_q‖², γ = u_p · u_q. Skip criterion `|γ| ≤ ε·√α·√β` checks that the cosine of the inter-column angle is below ε — confirmed as the standard Drmac-Veselic criterion. Zero-column pairs (`α=0 || β=0`) are skipped before the division by γ.

Rutishauser rotation: ζ = (β−α)/(2γ), then t = sign(ζ)/(|ζ|+√(1+ζ²)). The large-|ζ| branch `t = sign(ζ)/(|ζ|(1+√(1+1/ζ²)))` avoids ζ² overflow. c = 1/√(1+t²), s = ct. Column rotations u_p ← cu_p−su_q, u_q ← su_p+cu_q are correct for one-sided Jacobi (right-multiply by the Jacobi rotation J). Identical rotation applied to V. All correct.

Extraction: singular value σⱼ = ‖column j‖, then normalize. Selection sort descending over (σ, U columns, V columns). Default maxSweeps=30, default ε=fProxyZeroThreshold. Correct.

---

### 4. Symmetric Jacobi Eigenvalue Decomposition

**File:** `OP/Eigen.fProxy.cs`  
**Algorithm:** Classical cyclic Jacobi (all (p,q) pairs per sweep)  
**References:**
- Golub & Van Loan §8.4.3
- Rutishauser (1966), "The Jacobi Method for Real Symmetric Matrices"

**Verdict: MATCHES**

Convergence skip: `|A[p,q]| ≤ ε·0.5·(|A[p,p]|+|A[q,q]|)` — standard relative criterion. Rotation: θ = (A[q,q]−A[p,p])/(2·A[p,q]), t = sign(θ)/(|θ|+√(1+θ²)) (same formula as SVD, with the large-|θ| branch for overflow safety). Update formulas A[p,p] = A[p,p]−t·A[p,q], A[q,q] = A[q,q]+t·A[p,q], A[p,q]=A[q,p]=0 are the standard Jacobi identities derived from the trigonometric update after zeroing the off-diagonal element.

The off-diagonal sweep for rows i≠p,q: `newAip = c·aip−s·aiq`, `newAiq = s·aip+c·aiq`, applied symmetrically to both rows and columns. V accumulates the rotation in the same way. Eigenvalues extracted from the (approximately) diagonal A, then selection-sorted descending. Correct.

**Power iteration** (`powerIteration`): Standard Rayleigh-quotient iteration. Seed-on-zero guard uses deterministic values 1+(i&3). Residual check: ‖Av−λv‖∞ ≤ tol·max(1,|λ|). Correct.

---

### 5. Francis Double-Shift QR (General Eigenvalues)

**File:** `OP/Eigen.fProxy.cs`, method `eigenvaluesQR`  
**Algorithm:** EISPACK `elmhes` (Hessenberg reduction) + `hqr` (Francis double-shift)  
**References:**
- EISPACK `hqr.f`, netlib.org/slatec/lin/hqr.f (confirmed exact match)
- Golub & Van Loan §7.5
- Press et al., *Numerical Recipes* §11.6

**Verdict: MATCHES**

**Hessenberg reduction** (`elmhes`): Gaussian elimination with partial pivoting as a sequence of similarity transforms (row swap + column swap). Subdiagonal multipliers are stored in A[i,m-1] during elimination, then zeroed in a cleanup pass. Correct.

**Francis QR sweep** (`hqr`): Convergence test `math.abs(A[l,l-1]) + s0 == s0` (s0 = max(|A[l-1,l-1]|+|A[l,l]|, anorm)) is the floating-point addition test from EISPACK — correctly detects negligible subdiagonals.

Single real root (l==nn): eigenvalues[nn] = A[nn,nn] + t (accumulated shift). Complex pair (l==nn-1): p = 0.5·(A[nn-1,nn-1]−A[nn,nn]), q = p²+A[nn,nn-1]·A[nn-1,nn], z = sqrt(|q|). Real pair: z = p + copysign(z,p), eigenvaluesReal[nn] = x + (z≠0 ? x-w/z : x+z) (numerically stable avoidance of z=0 divide). Complex pair: real = x+p, imag = ±z. All correct.

**Exceptional shifts** at its=10 and its=20: `x = y = 0.75·s1`, `w = -0.4375·s1²`, `s1 = |A[nn,nn-1]|+|A[nn-1,nn-2]|`. Verified against EISPACK `hqr.f` source — exact match.

**`copysign` helper**: `b >= 0 ? |a| : -|a|`. Matches the EISPACK `SIGN(a,b)` function exactly.

---

### 6. Conjugate Gradient Solver

**File:** `OP/Solvers.fProxy.cs`  
**Algorithm:** Standard CG (Hestenes-Stiefel 1952)  
**References:**
- Hestenes & Stiefel (1952), "Methods of Conjugate Gradients for Solving Linear Systems"
- Golub & Van Loan §11.3.1

**Verdict: MATCHES**

Update order: r=b−Ax, p=r, rsold=r·r, then per iteration: Ap=A·p, α=rsold/(p·Ap), x+=αp, r−=αAp, rsnew=r·r, β=rsnew/rsold, p=βp+r, rsold=rsnew. This is the exact Hestenes-Stiefel CG order.

Convergence check: `rsnew ≤ tolerance²·(b·b)` — relative residual criterion ‖r‖/‖b‖ ≤ tol. Standard.

Breakdown guard: `!(pAp > 0)` returns false — NaN-safe (also catches NaN) and correctly detects non-SPD or numerical breakdown.

Zero-RHS shortcut: copies b to x (sanitizes NaN initial guess). Correct.

Pre-convergence check before the loop (rsold ≤ threshold): allows returning immediately if the initial x is already a solution. Correct.

---

### 7. FFT — Cooley-Tukey Radix-2 DIT

**File:** `OP/FFT.fProxy.cs`  
**Algorithm:** In-place radix-2 DIT (Decimation-In-Time) FFT  
**References:**
- Cooley & Tukey (1965), "An Algorithm for the Machine Calculation of Complex Fourier Series"
- Brigham, *The Fast Fourier Transform and Its Applications*

**Verdict: MATCHES**

Forward convention: X[k] = Σ x[n]·exp(−2πi·kn/N) (standard engineering / MATLAB convention, negative exponent). Bit-reversal permutation is the standard incremental method. Butterfly stages: per-stage twiddle factor `w = exp(−2πi/len)` (forward), accumulated as `curRe += ...` (complex multiplication). Correct.

Inverse via conjugate trick: conjugate input, apply forward FFT, conjugate output, scale by 1/N. This is a correct and well-known identity.

O(N²) DFT fallback: correctly handles arbitrary N. The precision caveat for float at large N is documented: "at N≈1e³ the angle's ulp approaches a radian." No bug, just a float precision limitation inherent to the algorithm.

---

### 8. Box-Muller Gaussian Sampler

**File:** `OP/RandomOP.fProxy.cs`, `fProxyGaussian`  
**Algorithm:** Box-Muller transform (Box & Muller 1958)  
**References:**
- Box & Muller (1958), "A Note on the Generation of Random Normal Deviates," *Ann. Math. Statist.*

**Verdict: MATCHES**

```
u1 = 1 - rng.NextFProxy()   // maps [0,1) to (0,1] — avoids log(0)
u2 = rng.NextFProxy()        // [0,1) is fine for trig argument
r  = sqrt(-2·log(u1))
z0 = r · cos(2π·u2)          // returned first
z1 = r · sin(2π·u2)          // cached as 'spare'
```

Correct per the original Box-Muller formula. The `1 - u` transform is the standard guard against log(0). The spare (sin branch) is stored fully scaled (mean+std applied) before caching, so a mid-fill change to mean/std cannot rescale a pending value — a correctness refinement.

---

### 9. Window Functions (DSP)

**File:** `OP/GenOP.fProxy.cs`, `window()`  
**References:**
- Harris (1978), "On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform"
- Oppenheim & Schafer, *Discrete-Time Signal Processing*, §7.3

**Verdict: MATCHES — All coefficients exact**

| Window   | Implementation                                              | Standard formula                                          |
|----------|-------------------------------------------------------------|-----------------------------------------------------------|
| Hann     | 0.5·(1 − cos(2πn/(N−1)))                                   | 0.5·(1 − cos(2πn/(N−1))) ✓                               |
| Hamming  | 0.54 − 0.46·cos(2πn/(N−1))                                 | 0.54 − 0.46·cos(2πn/(N−1)) ✓                             |
| Blackman | 0.42 − 0.5·cos(2πn/(N−1)) + 0.08·cos(4πn/(N−1))          | 0.42 − 0.5·cos(2πn/(N−1)) + 0.08·cos(4πn/(N−1)) ✓       |

All three use the `(N−1)` denominator (symmetric/periodic-extension convention). The Box type returns all-ones. N=1 degenerates gracefully to {1} for all types. Correct.

---

### 10. Householder QR and Column-Pivoted QR (Businger-Golub)

**File:** `OP/OrthoOP.fProxy.cs`  
**Algorithm:** Householder QR; QRCP (Businger & Golub 1965)  
**References:**
- Golub & Van Loan §5.1–5.4
- Businger & Golub (1965), "Linear Least Squares Solutions by Householder Transformations," *Numer. Math.*

**Verdict: MATCHES**

**Sign convention** (`genHouseholderPete`): After normalizing x by ‖x‖, adds sign(x[k]) to u[k]. This is the standard sign choice (same sign as x[k]) that avoids catastrophic cancellation when x[k] ≈ ‖x‖. Confirmed correct by Overton (2023) reference.

**Non-standard normalization**: The Householder vector u is scaled so that u^T·u = 2 (rather than the Golub-Van Loan convention u[0]=1). This means the reflection is applied as A ← A − u·(u^T·A) without the usual 2/‖v‖² factor. The math is equivalent: with u^T·u = 2, H = I − u·u^T is orthogonal. This is a valid and efficient normalization.

**Q reconstruction**: Backward accumulation of stored Householder vectors, applied in reverse order to I (built piece-by-piece with Q[i,d] = i==d?1:0 before each reflector). Correct.

**QRCP**: Among trailing columns d..n-1, the one with the largest partial 2-norm (over rows d..m-1) is swapped to position d. Uses exact recompute of partial norms (O(n²m) total) rather than the LAPACK-style cheap downdate. This sidesteps the known catastrophic-cancellation failure mode of norm downdating near rank-deficiency, at the cost of an n-fold constant in the pivot-selection inner loop. Numerically superior for the modest matrices this library targets.

**QRCP rank detection**: R diagonal is non-increasing; rank r = count of |R[i,i]| > tol where tol = relTol·|R[0,0]|, default relTol = max(m,n)·fProxyZeroThreshold. Matches SVD.pinvSolve convention. Correct.

---

### 11. Catmull-Rom Resampling

**File:** `OP/ResampleOP.fProxy.cs`  
**Algorithm:** Catmull-Rom cubic (Keys a=−0.5)  
**References:**
- Keys (1989), "Cubic Convolution Interpolation for Digital Image Processing," *IEEE Trans. ASSP*
- Catmull & Rom (1974), "A class of local interpolating splines"

**Verdict: MATCHES**

Four-point Catmull-Rom formula (t = fractional position, p0..p3 = four consecutive samples):
```
0.5·(2·p1 + (−p0+p2)·t + (2·p0−5·p1+4·p2−p3)·t² + (−p0+3·p1−3·p2+p3)·t³)
```
Confirmed correct by web research (matches Wikipedia, Keys reference, CMU spline notes exactly). This is the a=−0.5 Keys kernel. Reproduces linear polynomials exactly, quadratics exactly, cubics by construction.

Edge modes (Clamp, Wrap, Mirror) applied via `idx()` before lookup. Mirror uses no-edge-repeat convention `period = 2·(n−1)`, correct for symmetric reflection without duplicating endpoint.

Endpoint-pinning in `resampleInto`: `dst[0]=src[0]`, `dst[N-1]=src[N-1]` avoids FP drift at i·scale when i=(dstN-1) and scale is not a dyadic rational. Correct.

---

### 12. k-means++ Seeding

**File:** `ML/KMeans.fProxy.cs`, `SeedKMeansPlusPlus`  
**Algorithm:** Arthur & Vassilvitskii k-means++ (2007)  
**References:**
- Arthur & Vassilvitskii (2007), "k-means++: The Advantages of Careful Seeding," *SODA*

**Verdict: MATCHES**

Seeding: first centroid is a uniformly random point. Subsequent centroids c_i are drawn with probability proportional to D²(x) = min_{j<i} ‖x − c_j‖² (squared distance to nearest existing centroid). Confirmed by web research: D² weighting is the defining property of k-means++ and the theoretical guarantee for O(log k)-approximation.

Implementation uses incremental D² update (after placing c_i, update D2Weights[n] = min(D2Weights[n], dist²(x_n, c_i))), avoiding the O(k²Nd) recompute. Correct and efficient.

All-identical-points fallback: if total weight = 0, falls back to uniform random selection. Correct.

Empty-cluster reseed in Lloyd loop: assigns the farthest point to the empty cluster, marks it so subsequent empty-cluster rescans don't pick the same point (D2Weights[farthestPt] = -1). Correct.

Final sync: on all exit paths (convergence or maxIter), recomputes assignment and inertia from the final centroids. Ensures output consistency. Correct.

---

### 13. Gallery Test Matrices

**Files:** `Arena/Gallery.SPD.fProxy.cs`, `Gallery.Special.fProxy.cs`, `Gallery.Phase2.fProxy.cs`  
**References:**
- MATLAB `gallery()` documentation; Higham test matrix toolbox
- Moler (2014), "The Rosser Matrix" (Cleve's Corner, MathWorks)

**Hilbert**  
Formula: H[i,j] = 1/(i+j+1) (0-indexed). MATCHES standard. Trace = Σ1/(2i+1), severely ill-conditioned.

**Pascal**  
Recurrence P[i,j] = P[i-1,j] + P[i,j-1] with P[k,0]=P[0,k]=1 builds C(i+j,i). Integer-exact and overflow-safe. MATCHES.

**Lehmer**  
L[i,j] = (min(i,j)+1)/(max(i,j)+1) (0-indexed). MATCHES standard.

**MinIJ**  
A[i,j] = min(i,j)+1. MATCHES. Inverse is the (−1,2,−1) tridiagonal with last diagonal = 1.

**KMS (Kac-Murdock-Szego)**  
A[i,j] = ρ^|i−j|. MATCHES. SPD for |ρ|<1.

**Pei**  
αI + J (rank-1 update). Eigenvalues α+n (×1) and α (×n-1). MATCHES.

**Moler**  
A[i,j] = min(i,j)·α² + (i==j?1:α). Derived from UᵀU. MATCHES.

**Laplacian1D**  
Strang 2nd-difference tridiagonal (2, −1, −1). Eigenvalues 2−2cos(kπ/(n+1)). MATCHES.

**Clement**  
Superdiagonal: e[i] = √((i+1)·(n-1-i)) for 0-indexed i=0..n-2. Subdiagonal: e[i] = √(i·(n-i)). Confirmed correct by web research. Eigenvalues {n-1, n-3, …, -(n-1)}. MATCHES.

**Wilkinson W+**  
Diagonal: |m−i| where m=(n-1)/2. Off-diagonal: 1. n must be odd. Near-equal top eigenvalues. MATCHES.

**Fiedler**  
F[i,j] = |i−j|. One positive eigenvalue, n-1 negative. det = (−1)^(n−1)·(n−1)·2^(n-2). MATCHES.

**DingDong**  
Library formula: `0.5 / ((n − i − j) − 0.5)` = `0.5/(n-i-j-0.5)`.  
Canonical MATLAB formula (0-indexed conversion): `0.5/(n-i-j-1.5)`.  
**These differ by +1 in the subtracted constant.** The library produces a symmetric Hankel matrix with the same qualitative property (eigenvalues in (−π/2, π/2)) but it is not the standard DingDong matrix as defined in MATLAB's gallery and Higham's toolbox. **See Findings Table.**

**Frank**  
F[i,j] = n−max(i,j) for i ≤ j+1, else 0. For n=3: [[3,2,1],[2,2,1],[0,1,1]]. MATCHES standard definition. det=1.

**Vandermonde**  
V[i,j] = nodes[i]^j. Row-based powers from col 0=1 to col n-1. MATCHES.

**Companion**  
C[i,n-1] = −coeffs[i]; C[i,i-1]=1 for i>0. Eigenvalues = polynomial roots. MATCHES.

**Hadamard (Sylvester-Walsh)**  
H[i,j] = (−1)^popcount(i&j). Confirmed correct by web research (standard Sylvester construction). H^T·H = n·I. MATCHES.

**Circulant**  
C[i,j] = c[(j-i) mod n]. Eigenvalues = DFT of c. MATCHES.

**Kahan**  
K[i,i]=s^i; K[i,j]=−c·s^i for j>i. Ill-conditioned; classic unpivoted QR counterexample. MATCHES.

**Triw**  
Upper triangular, 1 on diagonal, α above. det=1, all eigenvalues=1. MATCHES.

**Lauchli**  
(n+1)×n: row 0 all-ones; rows 1..n are ε·I. Rank stress test. MATCHES.

**Cauchy (Gallery)**  
C[i,j]=1/(x[i]+y[j]) with division-by-zero guard. MATCHES. Note Hilbert is a special case (x[i]=y[i]=i+0.5 gives 1/(i+j+1)).

**GCD**  
A[i,j] = gcd(i+1,j+1). SPD, det = Πφ(k) (Smith's theorem). MATCHES.

**Redheffer**  
R[i,j] = 1 if j==0 or (j+1)%(i+1)==0. det = Mertens M(n). M(1..8)={1,0,-1,-1,-2,-1,-2,-2} per comment. MATCHES.

**Magic (Siamese)**  
de la Loubère filling. For n=3: [[8,1,6],[3,5,7],[4,9,2]]. All row/col/diagonal sums = n(n²+1)/2. MATCHES.

**Rosser**  
Hardcoded 8×8 symmetric matrix. Library eigenvalue comment:  
{−1020.0532, −0.1705, 0.2180, 999.9469, 1000.1207, 1019.5244, 1019.9936, 1020.4202}  
Confirmed CORRECT by web research (Moler 2014, MATLAB `rosser` documentation). Trace = 4040 ✓.  
**Note:** The audit-prompt cross-check values {−10.00274, 0.09824, 1, 1, 2, 2, 14.90194, 1020} are wrong — those do not correspond to the standard Rosser matrix.

**Parter**  
P[i,j] = 1/(i−j+0.5) (0-indexed). MATCHES. Denominator always non-zero half-integer. Singular values cluster near π.

**Prolate**  
A[i,j] = 2w (k=0) or sin(2πwk)/(πk) (k=|i-j|≥1). MATCHES. Eigenvalues in (0,1), requires 0<w<0.5.

**Grcar**  
G[i,j]=1 if d=j-i ∈ {0,1..k}; −1 if d=−1. Nonsymmetric banded Toeplitz. MATCHES.

**Lotkin**  
Row 0 = all-ones; row i≥1: 1/(i+j+1). Nonsymmetric, severely ill-conditioned. MATCHES.

---

### 14. Statistics Operations

**File:** `Statistics/StatsOP.fProxy.cs`  
**References:**
- Press et al., *Numerical Recipes*, §14 (basic statistics)
- NumPy documentation (percentile linear interpolation)

**Verdict: MATCHES**

- `variance`: Population variance (÷N). Explicit design choice; sample variant is `varianceSample` (÷(N-1)).
- `covariance`: Sample covariance (÷(M-1), Bessel-corrected). Correct for the columns=variables, rows=observations convention.
- `median`: Correct even/odd branching (sort + middle element or average of two).
- `Percentile` helper: `pos = p*(n-1)`, linear interpolation between sorted[lo] and sorted[hi]. Matches NumPy's `'linear'` method. Correct.
- `softmax`: Numerically stable (subtract row/column max before exp). Correct.
- `standardize`: Uses population std dev (÷N). Documented design choice.
- `correlation`: Pearson off-diagonal = C[i,j]/(s_i·s_j), clamped to [−1,1] to suppress FP overshoot. Zero-variance column sets off-diagonal to 0, diagonal to 1 by convention. Correct.

---

### 15. Probability Distributions

**File:** `OP/RandomOP.fProxy.cs`  
**References:** Standard probability texts (e.g., Devroye, *Non-Uniform Random Variate Generation*)

**Verdict: MATCHES — All ICDFs textbook-correct**

| Sampler        | ICDF Formula                             | Correct? |
|----------------|------------------------------------------|----------|
| Uniform        | min + (max−min)·u                        | ✓        |
| Exponential    | −log(1−u)/λ                             | ✓        |
| Rayleigh       | σ·√(−2·log(1−u))                        | ✓        |
| Weibull        | λ·(−log(1−u))^(1/k)                     | ✓        |
| Cauchy         | x₀ + γ·tan(π·(u−0.5)), clamped to avoid tan(±π/2) | ✓ |
| Logistic       | μ + s·log(u/(1−u)), clamped to avoid log(0)  | ✓   |
| Pareto         | x_m / (1−u)^(1/α)                       | ✓        |
| Triangular     | Piecewise sqrt formula with breakpoint at fc=(c−a)/(b−a) | ✓ |

All use `uc = 1−u` to map [0,1)→(0,1] for log-containing formulas. All guards (Cauchy/Logistic clamping to ε) are correct.

---

### 16. Multivariate Normal and Random Matrix Generation

**File:** `OP/RandomMatrixOP.fProxy.cs`  
**References:**
- Golub & Van Loan §2.5 (Cholesky sampling)
- Mezzadri (2007), "How to Generate Random Matrices from the Classical Compact Groups"
- Stewart (1980), "The Efficient Generation of Random Orthogonal Matrices with an Application"

**Verdict: MATCHES**

`multivariateNormalInpl`: z ~ N(0,I) via Box-Muller, dest = cholL·z + mean. Standard. Correct.

`randomOrthogonalInpl` (Haar measure): Fill n×n with N(0,1), QR-decompose, apply Haar sign fix (multiply column i of Q by sign(R[i,i])). Without the sign fix, Householder QR's Q is not uniformly distributed over O(n). Mezzadri (2007) is cited correctly. Correct.

`randomSpdInpl`: A = Q·Λ·Qᵀ where Q is Haar-uniform orthogonal and λᵢ ~ Uniform(minEig,maxEig). Qᵀ is computed before scaling Q's columns (important: scaling Q then transposing would give (QΛ)ᵀ = ΛQᵀ, not Qᵀ). Exact symmetry enforced via (A+Aᵀ)/2. Correct.

`randomMatrixWithConditionInpl`: U·Σ·Vᵀ where U,V are independent Haar-uniform orthogonal, and σᵢ = cond^(1−i/(k−1)) (logarithmic spacing). k=1 special case (σ=1). Correct.

`randomMatrixWithRankInpl`: dest = A·B, A~N(0,1)^(m×rank), B~N(0,1)^(rank×n). Has rank exactly `rank` with probability 1. Correct.

---

## Findings Table

| Severity | Area | Issue | Reference | Suggested Action |
|----------|------|-------|-----------|------------------|
| Medium | Gallery / DingDong | `fProxyDingDong`: library uses `0.5/(n−i−j−0.5)` (0-indexed) but the standard MATLAB/Higham DingDong uses `0.5/(n−i−j−1.5)` (0-indexed, i.e., `0.5/(n−i−j+0.5)` in 1-indexed). The two matrices have the same Hankel structure but different eigenvalue spectra. Users comparing against canonical DingDong eigenvalue tables will see discrepancies. | MATLAB `gallery('dingdong')`, Higham test matrix toolbox | Change denominator to `(fProxy)(n−i−j) + (fProxy)0.5` (i.e., `n−i−j+0.5`) to match the 1-indexed MATLAB formula, or add a doc note clarifying the index convention used. |
| Low | Cholesky / Pivoted | `choleskyDecompositionPivot`: `absScale` is computed as max over ALL entries of W (off-diagonal included), while LAPACK `dpstrf` scans only the initial diagonal. For genuine PSD input the results are identical (diagonal dominates). For indefinite input the library's scan is slightly more conservative (makes stopTol slightly larger, so it may stop at a higher effective rank). Not a correctness bug. | LAPACK `dpstrf.f`, line starting `AJJ = A(J,J)` | No change required; current behaviour is more defensive. Add a comment noting the deliberate deviation from LAPACK's diagonal-only scan. |
| Low | FFT / DFT O(N²) | `dft()` precision caveat for float at large N is documented ("at N≈1e³ the angle's ulp approaches a radian") but is in the `<summary>` only. Users in hot paths may miss it. | Standard FP analysis of trigonometric summation error | Consider adding an `#if UNITY_ASSERTIONS` runtime check that warns when float DFT is called with N > 512, or document the threshold more prominently at call site. |
| Info | Statistics | `variance` / `standardize` use population divisor (÷N), not sample (÷N-1). This is a design choice (sample variant exists as `varianceSample`), but callers used to NumPy default (÷N−1) may use the wrong function. | NumPy documentation (`ddof=0` vs `ddof=1`) | No bug. Consider adding a one-line note in `variance`'s summary saying "population (÷N); for sample (÷N−1) use `varianceSample`." |
| Info | Eigen / QR | `eigenvaluesQR` convergence guard `if (l < 0) l = 0;` is a no-op (the for-loop `for (l=nn; l>=1; l--)` leaves l≥0 on exit). The guard was copied verbatim from EISPACK/NR Fortran-to-C translation where the loop ran l=nn down to 0 (inclusive). | EISPACK `hqr.f` | No functional impact. Can be removed for clarity, or kept as a defensive guard against future refactors. |
| Info | Gallery / Rosser | The audit prompt's cross-check eigenvalues {−10.00274, 0.09824, 1, 1, 2, 2, 14.90194, 1020} are incorrect for the standard Rosser matrix. The library's comment {−1020.0532, −0.1705, 0.2180, 999.947, 1000.121, 1019.524, 1019.994, 1020.420} is correct (verified against Moler 2014, MATLAB documentation). | Moler (2014), MathWorks blog | No action on library. Correct the audit-prompt reference if reused. |
| Info | QRCP | Exact partial-norm recompute instead of LAPACK-style cheap downdate is O(n) times more work per pivot step but numerically superior near rank-deficiency. Trade-off is correct for the targeted matrix sizes. | Businger & Golub (1965); LAPACK `xGEQPF` vs `xGEQP3` | No change required for correctness. Document that exact recompute is a deliberate robustness choice. |
