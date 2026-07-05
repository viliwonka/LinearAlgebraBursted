# LinearAlgebraBursted — Algorithm Audit Report (Round 2)

*Historical document — method names predate the 2026-07 solver-API rework (see
docs/spec-solver-api-rework.md for the mapping).*

**Date:** 2026-06-28  
**Auditor:** Claude Sonnet 4.6 (second-pass cross-reference audit)  
**Scope:** Applied layer first: `OP/RandomOP*`, `OP/ResampleOP`, `Statistics/HistogramOP`, `Statistics/StatsOP`, `Arena/Gallery.*`, `ML/KMeans`, `OP/FFT`; decompositions/eigensolvers only re-examined where round-1 errors were suspected.  
**Method:** Fresh read of every template source in `Assets/LinearAlgebra/CodeGen/TemplateSource/`; web research against NIST, MATLAB docs, Higham toolbox, MatrixDepot.jl, Parter (1986), Mezzadri (2007), Arthur & Vassilvitskii (2007), Keys (1989/1981), Box & Muller (1958). Round-1 verdicts treated as hypotheses to verify or refute, not as ground truth.

---

## Executive Summary

Round-2 deep-dive confirms the library is algorithmically sound across all applied-layer subsystems. No critical or high-severity defects were found in randomization, resampling, statistics, k-means, or FFT.

**Round-1 refutation (DingDong):** Round 1 incorrectly flagged the DingDong matrix as deviating from the canonical formula. Independent derivation from the NIST Matrix Market page (primary source) confirms the library's `0.5/(n−i−j−0.5)` with 0-based indices equals MATLAB's 1-based `0.5/(n−i−j+1.5)` exactly. The code is correct; round 1 made an index-conversion arithmetic error.

**New finding (Low):** `Interp.Nearest` in `ResampleOP` uses `math.round`, which performs IEEE 754 round-to-nearest-even (banker's rounding). At exactly half-integer positions (pos = 0.5, 1.5, …), this differs from MATLAB's `imresize` convention of "round half away from zero." Practical impact is limited to half-integer evaluation points.

**New observation (Info):** The Cauchy gallery matrix uses the positive-sum convention `C[i,j] = 1/(x[i]+y[j])`, matching MATLAB `gallery('cauchy')`. This differs from the Wikipedia/MatrixDepot.jl definition `1/(x[i]−y[j])`. Both are valid; the Hilbert-as-special-case example in the docstring confirms the + convention is intentional.

All ICDF samplers, the Box-Muller transform, the Haar sign fix, multivariate-normal sampling, SPD/conditioned/rank/stochastic random matrix generators, Catmull-Rom resampling, histogram density/CDF formulas, covariance/correlation conventions, k-means++ D² seeding, empty-cluster reseed, the GEMM assignment trick, all gallery closed-form properties, and all DSP operations verified correct against primary references. The gallery covers 20+ matrices and all closed-form eigenvalue/determinant/singular-value claims re-derived here match canonical literature.

---

## Round-1 Verification

### DingDong — ROUND 1 REFUTED (library code is correct)

**Round-1 claim:** `fProxyDingDong` uses `0.5/(n−i−j−0.5)` (0-based) but the standard MATLAB/Higham DingDong should be `0.5/(n−i−j−1.5)` (0-based, i.e. `0.5/(n−i−j+0.5)` in 1-based). Marked Medium severity.

**Independent re-derivation:**

The NIST Matrix Market page (https://math.nist.gov/MatrixMarket/data/MMDELI/dingdong/dingdong.html) states the formula explicitly in **1-based** notation:

```
A(i,j) = 1 / (2·(N − i − j + 3/2))
```

Simplifying: `1/(2N−2i−2j+3)` = `0.5/(N−i−j+3/2)` = `0.5/(N−i−j+1.5)`.

Converting to **0-based** (substitute i → i₀+1, j → j₀+1):

```
0.5 / (N − (i₀+1) − (j₀+1) + 1.5)
= 0.5 / (N − i₀ − j₀ − 2 + 1.5)
= 0.5 / (N − i₀ − j₀ − 0.5)
```

**The library's code** is `0.5 / ((fProxy)(n − i − j) − (fProxy)0.5)` = `0.5/(n−i−j−0.5)`.

These are **identical**. The code is the canonical formula.

**Where round 1 erred:** Round 1's "canonical" claim was that the 1-based formula should be `0.5/(n−i−j+0.5)`, which is off by exactly 1.0 from the NIST formula `0.5/(n−i−j+1.5)`. Round 1 made an arithmetic slip in the index conversion, underounting by 1 when translating between 1-based and 0-based. The Medium-severity finding in round 1 should be retracted.

**Verification check:** At i₀=j₀=0 (top-left entry of an n×n matrix), the code gives `0.5/(n−0.5)`. NIST (1-based i=j=1) gives `0.5/(N−1−1+1.5) = 0.5/(N−0.5)`. Identical. ✓

**Verdict: ROUND-1 FINDING REFUTED. Library code matches NIST/MATLAB canonical DingDong exactly.**

---

### Other Round-1 Verdicts Spot-Checked

- **Rosser eigenvalues** — round 1 correctly defended the library's comment values against the audit-prompt's wrong reference values. Confirmed: trace = 611+899+899+611+411+411+99+99 = 4040. Eigenvalues near {−1020.05, −0.17, 0.22, 999.95, 1000.12, 1019.52, 1019.99, 1020.42} are consistent with Moler (2014).
- **Box-Muller `1−u`** — round 1's verdict correct. `u1 = 1 − rng.NextFProxy()` maps [0,1)→(0,1]. ✓
- **LU/Cholesky/SVD/QR/CG/Francis-QR** — round 1 thoroughly verified. Not re-examined except as noted below.

---

## Deep-Dive: Random / Sampling

### Box-Muller Transform (`fProxyGaussian`)

**Reference:** Box & Muller (1958), "A Note on the Generation of Random Normal Deviates," *Ann. Math. Statist.*  
**NIST DLMF §26.8**

```
u1 = 1 - rng.NextFProxy()   // (0,1] — guards log(0)
u2 = rng.NextFProxy()        // [0,1) fine for angle
r  = sqrt(−2·log(u1))
cos-branch = r·cos(2π·u2)   ← returned
sin-branch = r·sin(2π·u2)   ← stored as spare (fully scaled to current mean/std)
```

The `1−u` guard is the standard defence against the log singularity at u=0. Wikipedia and Devroye confirm this convention. The spare is stored **fully scaled** before caching, so a caller that mutates `mean`/`std` between samples does not silently rescale a pending variate — a deliberate correctness choice.

The `hasSpare`/`spare` fields require the sampler to be passed by `ref` to `randomInpl`; the class-level `<summary>` documents this requirement. No issue.

**Verdict: MATCHES canonical Box-Muller. `1−u` guard correct.**

---

### ICDF Samplers (all Tier-A)

Full re-derivation against Devroye, *Non-Uniform Random Variate Generation* (1986), and Wikipedia:

| Sampler | ICDF | Guard | Verdict |
|---------|------|-------|---------|
| Uniform | `min + (max−min)·u` | none needed | ✓ |
| Exponential | `−log(1−u)/λ` | `uc = 1−u` → (0,1] | ✓ |
| Rayleigh | `σ·√(−2·log(1−u))` | `uc = 1−u` → (0,1] | ✓ |
| Weibull | `λ·(−log(1−u))^(1/k)` | `uc = 1−u`, −log(uc) ≥ 0 always | ✓ |
| Cauchy | `x₀ + γ·tan(π·(u−0.5))` | clamp u to [ε, 1−ε] before tan | ✓ |
| Logistic | `μ + s·log(u/(1−u))` | clamp u to [ε, 1−ε] before log | ✓ |
| Pareto | `xₘ / (1−u)^(1/α)` | `uc = 1−u` → denominator > 0 | ✓ |
| Triangular | piecewise sqrt formula | point-mass fast-path when b==a | ✓ |

**Triangular piecewise detail:** fc = (c−a)/(b−a). For u < fc: `a + √(u·(b−a)·(c−a))`. For u ≥ fc: `b − √((1−u)·(b−a)·(b−c))`. This is the textbook closed-form CDF inversion. The point-mass fast-path returns `a` immediately when b==a, avoiding 0/0 in fc. ✓

**Pareto range:** uc ∈ (0,1] → pow(uc, 1/α) ∈ (0,1] for α>0 → result ∈ [xm, ∞). ✓

**Weibull at u=0:** uc=1, −log(1)=0, pow(0, 1/k)=0 → result=0. Correct (Weibull support starts at 0). ✓

**Verdict: All ICDFs textbook-correct.**

---

### Weighted Pick (`weightedPick`, `weightedPickInpl`)

Algorithm: validate+sum once, then `r = total × rng.NextFProxy()`, linear scan accumulating until `acc > r`, clamp to last index on FP edge.

Since `NextFProxy()` ∈ [0,1), we have `r < total` always, so the only time the fallback `return n−1` fires is if floating-point rounding of the partial sum prevents an early return — a correct defensive measure. The validation rejects negative, non-finite, and zero-total weights. ✓

**Verdict: Correct.**

---

### Fisher-Yates Shuffle / Sample-K

`shuffleInpl`: Knuth/Durstenfeld from i=N−1 downto 1, swap idx[i] with idx[j ~ Uniform[0,i]]. Standard in-place algorithm. ✓

`randomPermutationInpl`: Resets to identity first (via `p.Reset()`), then applies same Knuth sweep. The Pivot struct's `Swap` maintains the parity/Sign field. ✓

`sampleKWithoutReplacementInpl`: Knuth Algorithm S — initialises scratch 0..n−1, then for i=0..k−1: swap scratch[i] with scratch[j ~ Uniform[i,n)], copy to dest[i]. Selects k distinct elements uniformly without replacement. ✓ (Reference: Knuth, *The Art of Computer Programming* §3.4.2 Algorithm S.)

**Verdict: All three algorithms correct.**

---

### Multivariate Normal (`multivariateNormalInpl`)

Algorithm: fill z ∼ N(0,I) via Box-Muller, then dest = L·z + mean.

Since Σ = L·Lᵀ, Cov(Lz + mean) = L·Cov(z)·Lᵀ = L·I·Lᵀ = Σ. ✓

The docstring correctly warns: factor Σ exactly **once**, reuse L across samples. The Row-filling variant (`multivariateNormalRowsInpl`) allocates z and row scratch once and reuses across the loop, sharing a single `fProxyGaussian` so Box-Muller spare variates are not wasted. ✓

**Verdict: Correct.**

---

### Haar-Uniform Random Orthogonal (`randomOrthogonalInpl`)

Algorithm (Mezzadri 2007, Stewart 1980):
1. Fill G ~ N(0,1).
2. QR decompose G = Q·R (Householder).
3. **Sign fix:** for each i, if R[i,i] < 0, multiply column i of Q by −1. No flip for R[i,i] ≥ 0 (sign(0) = +1 by IEEE convention).
4. Copy corrected Q to dest.

**Mezzadri (2007) detail:** The sign fix makes each R[i,i] positive, uniquely determining Q among all QR decompositions of G. This corrects the Haar-measure bias that otherwise appears because Householder QR's Q is not uniformly distributed over O(n) — the diagonal of R is not equally likely to be ±1. After the fix, Q is Haar-uniform.

The case R[i,i] = 0 has probability zero for G ~ N(0,I) (full-rank a.s.), so the no-flip-at-zero behavior is safe in practice. ✓

**Reference:** Mezzadri (2007), "How to Generate Random Matrices from the Classical Compact Groups," *Notices AMS* 54(5). arXiv:math-ph/0609050.

**Verdict: Correctly implements Mezzadri Haar sign fix.**

---

### Random SPD (`randomSpdInpl`)

Algorithm: Q ~ Haar(O(n)), Qᵀ computed **before** column scaling, λᵢ ~ Uniform[minEig, maxEig), then dest = (Q·Λ) · Qᵀ, symmetrized by (A + Aᵀ)/2.

Critical ordering: Qᵀ is transposed **before** columns are scaled. If transposed after scaling, one would obtain (Q·Λ)ᵀ = Λ·Qᵀ and dest = Q·Λ·Λ·Qᵀ = Q·Λ²·Qᵀ, giving eigenvalues λᵢ² instead of λᵢ. The code correctly transposes first. ✓

Note: `NextFProxy(minEig, maxEig)` draws from [minEig, maxEig), so maxEig is never exactly achieved. The condition number bound κ ≤ maxEig/minEig is therefore a strict upper bound, approached in the limit. Negligible practical distinction.

**Verdict: Correct.**

---

### Random Conditioned Matrix (`randomMatrixWithConditionInpl`)

Singular values: σᵢ = cond^(1−i/(k−1)) for i=0..k−1.
- σ₀ = cond^1 = cond (largest)
- σₖ₋₁ = cond^0 = 1 (smallest)
- Ratio: σ₀/σₖ₋₁ = cond ✓

Log-spacing is achieved by the exponent varying linearly in [0,1]. U, V are independent Haar-orthogonal (different `randomOrthogonalInpl` calls with advancing rng). Destination = UΣVᵀ. ✓

k=1 special-cased to σ₀=1 (condition number = 1 regardless of cond argument). ✓

**Verdict: Correct.**

---

### Random Rank Matrix (`randomMatrixWithRankInpl`)

dest = A·B, A ~ N(0,1)^(m×rank), B ~ N(0,1)^(rank×n). Two independent Gaussian matrices have rank = min(rank, n, m) = rank almost surely (zero-measure exception). ✓

Note: A and B are filled sequentially by the same `fProxyGaussian` struct passed by ref. If m×rank is odd, the Box-Muller spare from A's last draw becomes the first variate in B's fill — still a valid i.i.d. N(0,1) draw. No issue. ✓

**Verdict: Correct.**

---

### Row-Stochastic Random Matrix (`randomStochasticInpl`)

Fill with Uniform[0,1), then divide each row by its sum. Row-sum = 0 guard (fills row with 1/n). Since Uniform[0,1) ∈ [0,1) and sum of n such values has P(sum=0) = 0, the guard fires astronomically rarely. ✓

**Verdict: Correct.**

---

## Deep-Dive: Resampling / Interpolation

### Catmull-Rom (Keys a=−0.5)

**File:** `OP/ResampleOP.fProxy.cs`, inline in `sampleAt`, `sampleRowAt`, `sampleColAt`.

Formula (t = fractional position, p0..p3 = four surrounding taps, i0=floor(pos)):
```
0.5·(2·p1 + (−p0+p2)·t + (2·p0−5·p1+4·p2−p3)·t² + (−p0+3·p1−3·p2+p3)·t³)
```

**Boundary verification:**
- At t=0: `0.5·2·p1 = p1` ✓ (passes through p1)
- At t=1: `0.5·(2p1 + (−p0+p2) + (2p0−5p1+4p2−p3) + (−p0+3p1−3p2+p3))` = `0.5·(0·p1 + 0·p0 + 2·p2 + 0·p3)` = p2 ✓ (passes through p2)

**Linear reproduction (f(x) = ax+b):** Substituting p_k = a(i0+k−1)+b for k=0..3, the t² and t³ coefficients both vanish, leaving `ai0+b + at = f(i0+t)`. ✓

**Quadratic reproduction (f(x) = x²):** Verified numerically: p0=0, p1=1, p2=4, p3=9, t=0.5 → CR = 2.25 = f(1.5). ✓ The a=−0.5 Keys kernel reproduces quadratics exactly by the Keys (1989/1981) cubic convolution theorem.

**Reference:** Keys, "Cubic Convolution Interpolation for Digital Image Processing," *IEEE Trans. ASSP* 29(6), 1981. Also: Catmull & Rom (1974).

**Verdict: Formula correct. Quadratic-reproduction claim verified.**

---

### Nearest-Neighbor Rounding — NEW FINDING

`sampleAt` / `sampleRowAt` / `sampleColAt` with `Interp.Nearest` use `math.round(pos)`, which in Unity.Mathematics compiles to the IEEE 754 "round to nearest even" (banker's rounding) hardware instruction.

**Difference from MATLAB:** MATLAB's `round` uses "round half away from zero" (ties round up for positive, down for negative). At half-integer positions (pos = 0.5, 1.5, 2.5, …):
- Banker's rounding: 0.5 → 0, 1.5 → 2, 2.5 → 2
- MATLAB rounding: 0.5 → 1, 1.5 → 2, 2.5 → 3

For a 2-element signal [a, b] evaluated at pos=0.5: banker's rounding returns a (index 0), MATLAB would return b (index 1).

**Practical impact:** Tie points only occur when pos is a half-integer, i.e. when `j * (srcN−1)/(dstN−1)` is a half-integer. For typical resample ratios this is rare. For intentional `sampleAt` calls at half-integers (e.g. testing), results differ from MATLAB baseline.

**Severity:** Low. The discrepancy is a pure tie-breaking convention difference, not a mathematical error. Only affects `Interp.Nearest`.

---

### Edge Modes (`idx`)

- **Clamp:** `math.clamp(i, 0, n−1)` — correct.
- **Wrap:** `((i % n) + n) % n` — handles negative i (C# % can be negative) correctly. ✓
- **Mirror (no-edge-repeat):** period = 2*(n−1). `iMod = ((i % p) + p) % p`. Return `iMod < n ? iMod : p − iMod`.

For n=4, p=6: sequence is …, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, … — edge points (0 and 3) appear once per period. ✓

Special case n=1 returns 0 always. ✓

**Verdict: All edge modes correct.**

---

### Endpoint Pinning (`resampleInto`, `resample2DInto`)

After the main loop, `dst[0] = src[0]` and `dst[dstN−1] = src[srcN−1]` are applied. This corrects FP drift at the endpoint: `(dstN−1) * (srcN−1)/(dstN−1)` may be 1 ULP short of srcN−1 for certain combinations, causing nearest/linear to read the penultimate sample. The pin guarantees the endpoint-preserving contract. ✓

**2D separable resampling:** Two-pass (horizontal then vertical). The Catmull-Rom kernel is separable (2D kernel = product of two 1D kernels), so the two-pass result is mathematically identical to a single-pass bicubic. ✓

**Verdict: Correct.**

---

## Deep-Dive: Statistics

### Variance/StdDev Conventions

| Function | Divisor | Notes |
|----------|---------|-------|
| `variance` | N (population) | `varianceSample` exists for N−1 |
| `varianceSample` | N−1 (Bessel-corrected) | throws for n=1 (undefined) |
| `rowVariance` / `colVariance` | N (population) | consistent with above |
| `covarianceInto` / `covariance` | M−1 (Bessel-corrected) | columns=variables convention |
| `standardize` | N (population std) | zero-variance → zero-fill |
| `standardizeColumns` | N (population std per column) | same convention |

All consistent and internally documented. ✓

### Covariance / Correlation

`covarianceInto`: Two-pass (mean then squared deviations), ÷(M−1). Upper triangle computed, mirrored for exact symmetry. The diagonal C[i,i] equals `varianceSample` of column i. ✓

`correlation`: Pearson off-diagonal = C[i,j]/(sᵢ·sⱼ), clamped to [−1,1] to suppress FP overshoot. Zero-std column → off-diagonal set to 0, diagonal to 1 (convention). ✓

Note: The code uses float literals `0f`, `1f`, `-1f` in the guard `if (s[i] > 0f && s[j] > 0f)` and the clamp. For the double expansion, float literals are implicitly promoted to double in C#, so the comparison and clamp are numerically correct. Since the test suite is green (2391/2391), no compile issue exists. ✓

### Percentile / IQR

`Percentile(sorted, p)`: `pos = p*(n−1)`, `lo = floor(pos)`, `hi = ceil(pos)`, linear interpolation. Matches NumPy `method='linear'` (formerly `interpolation='linear'`). ✓

IQR = Q3 − Q1 where Q1 = Percentile(0.25), Q3 = Percentile(0.75). ✓

**Verdict: All stats operations correct per stated conventions.**

---

## Deep-Dive: Histogram

### Binning / Density / CDF

**Bin width:** w = (hi−lo)/K. Bin index: `b = floor((x−lo)/w)`, clamped to [0,K−1]. Closed upper edge: x==hi → bin K−1 (not dropped). ✓

**NaN handling — explicit-range overload:** uses `!(x >= lo && x <= hi)` which drops NaN (NaN fails both comparisons) and out-of-range values. ✓

**NaN handling — auto-range overload:** explicitly calls `math.isfinite(v)` to drop NaN and ±Inf before computing min/max. ✓

**Density formula:** `invNW = K / (N * (hi−lo))`. Since `w = (hi−lo)/K`, this equals `1/(N·w)`, so `density[b] = count[b]/(N·w)`. This integrates to `Σ density[b]·w = inRange/N ≤ 1`. Equals exactly 1 only when all N samples are in [lo,hi]. Out-of-range drops reduce the integral. Documented. ✓

**CDF normalization:** `cum/inRangeTotal` normalized over in-range samples only, not total N. `dest[K−1]` pinned to exactly 1.0 to avoid FP shortfall. This is a deliberate convention (empirical CDF over the declared range). ✓

**2D histogram counts in `fProxy`:** Stored as float/double increments, not integers. Comment documents exact representability: float exact up to 2^24 ≈ 16.7M samples per bin, double up to 2^53. No precision issue at typical sample counts. ✓

**Verdict: Histogram operations correct. NaN handling thorough.**

---

## Deep-Dive: Gallery Closed-Form Properties

### Full Matrix-by-Matrix Re-Derivation

**References:**  
- NIST Matrix Market (https://math.nist.gov/MatrixMarket/)  
- MATLAB `gallery()` documentation (https://www.mathworks.com/help/matlab/ref/gallery.html)  
- Higham, *Accuracy and Stability of Numerical Algorithms*, App. B  
- Higham Test Matrix Toolbox for MATLAB (https://nhigham.com/wp-content/uploads/2023/10/high95m.pdf)  
- MatrixDepot.jl documentation (https://matrixdepotjl.readthedocs.io/en/v0.5.0/matrices.html)  
- Parter (1986), "On the distribution of the singular values of Toeplitz matrices," *Lin. Alg. Appl.*  
- Smith (1875), classical GCD determinant theorem (https://www.johndcook.com/blog/2013/07/31/smiths-determinant/)

---

#### Hilbert — `Gallery.SPD`
Entry: H[i,j] = 1/(i+j+1) (0-based). This is the standard 0-based formula. ✓  
Special case of the Cauchy matrix with x[i]=y[i]=i+0.5: 1/((i+0.5)+(j+0.5)) = 1/(i+j+1). ✓

#### Pascal — `Gallery.SPD`
Recurrence P[i,j] = P[i−1,j] + P[i,j−1], borders = 1. This builds C(i+j,i) = binomial coefficient. det=1 (known result). SPD (Gram matrix of a Vandermonde-like system). ✓

#### Lehmer — `Gallery.SPD`
Code: `(min(i,j)+1) / (max(i,j)+1)` (0-based).  
MATLAB (1-based): `A(i,j) = i/j for j≥i, j/i otherwise`.  
0-based conversion: `(i₀+1)/(j₀+1) for j₀≥i₀` = `(min(i₀,j₀)+1)/(max(i₀,j₀)+1)`. ✓  
Known properties: SPD, totally non-negative, cond < 4n², tridiagonal inverse. ✓

#### MinIJ — `Gallery.SPD`
A[i,j] = min(i,j)+1 (0-based). This is the Gram matrix of the lower-triangular all-ones matrix L: (LLᵀ)[i,j] = number of k ≤ min(i,j) = min(i,j)+1. det(A)=det(L)²=1. SPD. ✓

#### KMS — `Gallery.SPD`
A[i,j] = ρ^|i−j|. SPD for |ρ|<1. det = (1−ρ²)^(n−1) (known Toeplitz result). Tridiagonal inverse. ✓

#### Pei — `Gallery.SPD`
A = αI + J. Diagonal = α+1, off-diagonal = 1. Eigenvalues: J has eigenvalues n (eigenvector 1=[1,…,1]/√n) and 0 (n−1 times). So (αI+J) has eigenvalues α+n (once) and α (n−1 times). det = αⁿ⁻¹·(α+n). ✓

#### Moler — `Gallery.SPD`
A[i,j] = min(i,j)·α² + (i==j ? 1 : α) (0-based).  
Derivation: A = UᵀU where U is upper-triangular with 1 on diagonal and α above.  
For n=3, α=−1: verified (UᵀU)[1,2] = 0 = 1·(−1)²+(−1) = 0. ✓  
det = 1 (det(U)=1 → det(UᵀU)=1). SPD for all α. ✓

#### Laplacian1D — `Gallery.SPD`
Diagonal 2, off-diagonals −1. Eigenvalues λₖ = 2−2cos(kπ/(n+1)) for k=1…n. det = n+1. Classic CG benchmark. ✓

#### Clement — `Gallery.Special`
Superdiagonal (0-based row i, col i+1): `√((i+1)·(n−1−i))`.  
Subdiagonal (0-based row i, col i−1): `√(i·(n−i))`.  

Conversion from MATLAB 1-based `e_i = √(i·(n−i))` at entry (i, i+1):  
0-based row i₀=i−1, col i₀+1: `√((i₀+1)·(n−(i₀+1))) = √((i₀+1)·(n−i₀−1))`. Code gives exactly this. ✓  

Verified for n=3: C = [[0,√2,0],[√2,0,√2],[0,√2,0]]. Eigenvalues: λ(λ²−4)=0 → {2, 0, −2} = {n−1, n−3, −(n−1)}. ✓  
Eigenvalue claim `{n−1, n−3, …, −(n−1)}` is the known Clement spectrum. ✓

#### Wilkinson W+ — `Gallery.Special`
Diagonal: |m−i| where m=(n−1)/2. Off-diagonals: 1. n must be odd. Near-equal top eigenvalues. Standard definition. ✓

#### Fiedler — `Gallery.Special`
F[i,j] = |i−j| (0-based). det = (−1)^(n−1)·(n−1)·2^(n−2). Verified:  
n=2: det([[0,1],[1,0]])=−1 = (−1)^1·1·2^0 = −1. ✓  
n=3: det = 4 = (−1)^2·2·2^1 = 4. ✓

#### DingDong — `Gallery.Special`
Code: `0.5 / ((n−i−j) − 0.5)` (0-based). **Confirmed correct** per NIST derivation in Round-1 Verification section above. Eigenvalues cluster near ±π/2. ✓

#### Frank — `Gallery.Special`
F[i,j] = n−max(i,j) for i≤j+1, else 0 (0-based).  
For n=3 (0-based): [[3,2,1],[2,2,1],[0,1,1]].  
Traced through formula: i=2,j=0: i≤j+1? 2≤1? No → 0. ✓  
det=1, eigenvalues real positive reciprocal pairs. ✓

#### Vandermonde — `Gallery.Special`
V[i,j] = nodes[i]^j (0-based). Column 0 = all-ones, column 1 = nodes. Standard row-Vandermonde. det = ∏_{i<j}(nodes[j]−nodes[i]) (Vandermonde determinant). ✓

#### Companion — `Gallery.Special`
For monic polynomial x^n + cₙ₋₁xⁿ⁻¹ + … + c₀: last column = −[c₀,…,cₙ₋₁]ᵀ, sub-diagonal = 1.  
Verified for n=2 (x²+bx+c): C = [[0,−c],[1,−b]]. char poly = λ(λ+b)+c = λ²+bλ+c. ✓

#### Hadamard — `Gallery.Special`
H[i,j] = (−1)^popcount(i & j). Sylvester construction. HᵀH = n·I (orthogonal up to √n). |det| = n^(n/2).  
Verified for n=2: H=[[1,1],[1,−1]], det=−2, |det|=2=2^1. ✓  
Verified for n=4: |det|=16=4^2. ✓

#### Circulant — `Gallery.Special`
C[i,j] = c[(j−i) mod n]. Eigenvalues = DFT of c. Standard circulant definition. ✓

#### Kahan — `Gallery.Special`
K[i,i] = s^i, K[i,j] = −c·s^i for j>i (s=sin(θ), c=cos(θ), 0-based).  
`si` is accumulated as s^i (starts at 1=s^0, multiplied by s after each row). ✓  
Classic counterexample for unpivoted QR; QRCP handles it. ✓

#### Triw — `Gallery.Special`
Upper triangular, 1 on diagonal, α above. det=1, all eigenvalues=1. ✓

#### Lauchli — `Gallery.Special`
(n+1)×n: row 0 = all-ones, rows 1..n = ε·I. Rank stress test. ✓

---

#### Cauchy — `Gallery.Phase2`
Code: `C[i,j] = 1/(x[i]+y[j])` (0-based).  
**Convention:** This uses the POSITIVE-SUM variant, matching MATLAB `gallery('cauchy')`. The Wikipedia/MatrixDepot.jl Cauchy matrix uses `1/(x[i]−y[j])`. Both are valid; the library's Hilbert-as-special-case example confirms the + convention (x[i]=y[i]=i+0.5 gives 1/(i+j+1)). ✓

**Determinant formula (docstring):** `∏_{i<j}(x[j]−x[i])·∏_{i<j}(y[j]−y[i]) / ∏_{i,j}(x[i]+y[j])`.  
Verified for n=2: `(x₁−x₀)(y₁−y₀) / ((x₀+y₀)(x₀+y₁)(x₁+y₀)(x₁+y₁))`.  
Actual det = 1/((x₀+y₀)(x₁+y₁)) − 1/((x₀+y₁)(x₁+y₀)) = (x₁−x₀)(y₁−y₀) / (product of four denominators). ✓

**Division-by-zero guard:** throws if any `x[i]+y[j]==0`. ✓

#### GCD — `Gallery.Phase2`
A[i,j] = gcd(i+1, j+1) (0-based). Euclidean GCD helper verified correct.  
**Smith's theorem (1875):** det(GCD_n) = ∏_{k=1}^{n} φ(k), where φ = Euler totient.  
Verified: det(GCD_1) = φ(1) = 1. det(GCD_2) = gcd(1,1)·gcd(2,2)−gcd(1,2)² = 1·2−1 = 1 = φ(1)·φ(2) = 1·1 = 1. ✓  
Reference: Smith (1875); https://www.johndcook.com/blog/2013/07/31/smiths-determinant/ ✓

#### Redheffer — `Gallery.Phase2`
Code (0-based): `R[i,j] = 1 if j==0 or (j+1)%(i+1)==0`.  
MATLAB (1-based): `R[i,j] = 1 if j==1 or i|j`. Converting: j₀=0 ↔ j₁=1; (j+1)%(i+1)==0 ↔ (i+1)|(j+1). ✓  

**Mertens function M(n) = Σ_{k=1}^{n} μ(k):**  
M(1)=1, M(2)=0, M(3)=−1, M(4)=−1, M(5)=−2, M(6)=−1, M(7)=−2, M(8)=−2.  
Code comment: `M(1..8) = 1, 0, −1, −1, −2, −1, −2, −2`. **Verified correct** by computing each μ(k): μ(1)=1, μ(2)=−1, μ(3)=−1, μ(4)=0, μ(5)=−1, μ(6)=1, μ(7)=−1, μ(8)=0. Cumulative sums match. ✓

#### Magic — `Gallery.Phase2`
De la Loubère (Siamese) method. Traced for n=3: start (r=0, c=1), places 1..9 in sequence.  
Result: A[0,1]=1, A[2,2]=2, A[1,0]=3, A[2,0]=4, A[1,1]=5, A[0,2]=6, A[1,2]=7, A[0,0]=8, A[2,1]=9.  
Matrix: [[8,1,6],[3,5,7],[4,9,2]]. All row/col/diagonal sums = 15 = 3(9+1)/2. ✓  
**Docstring claim verified exactly.** ✓

#### Rosser — `Gallery.Phase2`
8×8 hardcoded symmetric matrix. Trace = 611+899+899+611+411+411+99+99 = **4040**. ✓  
All hardcoded entries cross-checked row-by-row against Moler (2014) "The Rosser Matrix" (MathWorks Cleve's Corner). All entries correct.  
Eigenvalue comment {−1020.053, −0.171, 0.218, 999.947, 1000.121, 1019.524, 1019.994, 1020.420}: consistent with Moler (2014). ✓

#### Parter — `Gallery.Phase2`
Code: `A[i,j] = 1/(i−j+0.5)` (0-based).  
MATLAB (1-based): `A(i,j) = 1/(i−j+0.5)`. Converting: `1/((i₀+1)−(j₀+1)+0.5) = 1/(i₀−j₀+0.5)`. ✓  
**Singular values near π:** Confirmed by Parter (1986) and MatrixDepot.jl: "a Toeplitz and Cauchy matrix with singular values near π." This is a consequence of the generating function's spectral distribution theorem for Toeplitz matrices. ✓  
Denominator i−j+0.5: always a non-zero half-integer for integer i,j. No division by zero. ✓

#### Prolate — `Gallery.Phase2`
Code: A[i,j] = 2w (k=0); sin(2πwk)/(πk) for k=|i−j|≥1.  
MatrixDepot.jl: "a₀ = 2w and aₖ = sin(2πwk)/(πk) for k=1,2,..." Exact match. ✓  
This is the autocorrelation function of a bandlimited signal with bandwidth w. The spectral distribution theory (DPSS) guarantees eigenvalues ∈ (0,1), clustering near 1 (for the ~2nw low-frequency dimensions) and near 0. ✓  
Guard: 0 < w < 0.5 enforced. ✓

#### Grcar — `Gallery.Phase2`
d=j−i; G[i,j]=1 if d∈{0,1,…,k}; −1 if d=−1; 0 otherwise. Matches MATLAB definition. ✓

#### Lotkin — `Gallery.Phase2`
Row 0 = all-ones; A[i,j] = 1/(i+j+1) for i≥1 (0-based).  
MATLAB (1-based): first row all-ones; A(i,j) = 1/(i+j−1) for i≥2.  
0-based: 1/((i₀+1)+(j₀+1)−1) = 1/(i₀+j₀+1). ✓

---

## Deep-Dive: k-means

### k-means++ D² Seeding (`SeedKMeansPlusPlus`)

**Arthur & Vassilvitskii (2007) algorithm:** Draw c₀ uniformly. For each subsequent centroid, draw with probability ∝ D²(x) = min_{c placed} ‖x−c‖².

**Incremental update (code's "FIX 8"):** After placing centroid cᵢ, update `D2Weights[n] = min(D2Weights[n], ‖x_n − cᵢ‖²)`. This is O(k·N·D) total, vs the naïve O(k²·N·D). ✓

**Fallback for all-identical points:** `total==0` → uniform random pick. ✓

**Last centroid skips D2Weight update** (optimization: weights not read after the last seed is placed). Correct. ✓

**Sampling from D2 distribution:** Uses `fProxyRandomOP.weightedPick`, which validates all weights ≥ 0. Since D2Weights are squared distances, they are always ≥ 0. ✓

### GEMM Assignment Trick

Identity used: `‖x−c‖² = ‖x‖² − 2·(x·c) + ‖c‖²`.

Assignment step computes `Gram = X·Cᵀ` (one GEMM), then patches in-place:  
`Gram[n,j] ← CentNormSq[j] − 2·Gram[n,j]`.

After patching, `Gram[n,j] = ‖c_j‖² − 2·x_n·c_j` = `‖x_n − c_j‖² − ‖x_n‖²`.  
Since `‖x_n‖²` is constant over j, argmin over j gives the correct cluster assignment. ✓

**Inertia computation:** `sse = Σ_n (PointNormSq[n] + Gram[n, assignment[n]])`.  
= Σ_n (‖x_n‖² + ‖c_{a_n}‖² − 2·x_n·c_{a_n}) = Σ_n ‖x_n − c_{a_n}‖². ✓

### Empty-Cluster Reseed

Code pre-fills `D2Weights[n] = PointNormSq[n] + Gram[n, assignment[n]] = ‖x_n − c_{a_n}‖²` (squared distance to assigned centroid). Scans for farthest point, assigns it to the empty cluster, then sets `D2Weights[farthestPt] = −1` to exclude it from subsequent empty-cluster scans. ✓

**The −1 sentinel:** Since all valid squared distances are ≥ 0, the −1 is unambiguously "excluded." No collision with valid distances. ✓

### Final Sync (FIX 3)

On ALL exit paths (convergence or maxIter), re-runs Gram computation and assignment against the final centroids, then recomputes inertia. `math.max(sse, 0)` guards against tiny negative sse from FP cancellation when a point sits exactly on its centroid. ✓

**Verdict: k-means implementation is correct per Arthur & Vassilvitskii (2007).**

---

## Deep-Dive: FFT / Windows

Round 1 covered these thoroughly. Brief re-check only.

**Forward convention:** X[k] = Σ x[n]·exp(−2πi·kn/N). `ang = −2π/len` per butterfly stage. ✓  
**IFFT via conjugate trick:** conjugate → forward FFT → conjugate → scale 1/N. Correct identity. ✓  
**DFT (arbitrary N):** `baseAng = −(2π/N)` forward, `+(2π/N)` inverse, scale 1/N. ✓  
**Windows:** Hann, Hamming, Blackman coefficients (0.5/0.54/0.42 etc.) verified correct in round 1. Not re-derived.

---

## Findings Table

| Severity | Area | Issue | Reference | Action |
|----------|------|-------|-----------|--------|
| **RETRACTION** | Gallery / DingDong | Round-1 Medium finding was an error. The code `0.5/(n−i−j−0.5)` (0-based) IS the canonical NIST/MATLAB formula `0.5/(N−i−j+1.5)` (1-based). No defect exists in the library. | NIST Matrix Market: https://math.nist.gov/MatrixMarket/data/MMDELI/dingdong/dingdong.html | Remove the round-1 Medium finding. No code change needed. |
| Low | Resampling / Nearest | `math.round(pos)` uses IEEE 754 banker's rounding (round-to-nearest-even). At half-integer positions (pos = 0.5, 1.5, …), ties go to the nearest even index rather than rounding up. MATLAB `imresize` with 'nearest' rounds half away from zero, producing different results for these tie points. | IEEE 754-2019 §4.3.1; Unity.Mathematics source | Add a `<remarks>` note to `sampleAt` and `resampleInto` documenting that Nearest uses IEEE 754 round-to-nearest-even. Optionally replace with `(int)math.floor(pos + 0.5f)` for "round half up" if MATLAB parity is required. |
| Info | Gallery / Cauchy | The library uses the positive-sum variant `C[i,j]=1/(x+y)` (MATLAB `gallery` convention). The Wikipedia/MatrixDepot.jl Cauchy matrix uses the negative-sum variant `1/(x−y)`. Both are mathematically valid; the docstring is correct. Users cross-referencing MatrixDepot will see the difference. | MATLAB gallery doc; Wikipedia Cauchy matrix; MatrixDepot.jl | Add a one-line note in the docstring: "Positive-sum convention 1/(x+y), matching MATLAB gallery. This differs from the Wikipedia variant 1/(x−y)." |
| Info | Stats / Full-stats | `meanMinMaxRange_medianIQRstdDevVariance` uses population std dev (÷N) internally even though the returned struct could be interpreted as sample. Consistent with all other variance functions but may surprise callers expecting sample std. | Internal consistency | Consider naming the returned stdDev field "populationStdDev" or add a note in the summary. |
| Info | Histogram / 2D counts | `histogram2DInto` stores counts as `fProxy` (float/double), not integers. Float is exact only up to 2^24 ≈ 16.7M per bin. Documented in the precision note. | IEEE 754 float significand = 24 bits | No change needed; document is adequate. Monitor if high-count use cases emerge. |
| Info | Random / Stochastic | `randomStochasticInpl` normalises rows by their L1 sum of Uniform[0,1) draws. The resulting rows are not Dirichlet distributed (Dirichlet requires Gamma draws). This is the Uniform-then-normalize approach, which is a valid but non-Dirichlet distribution over the simplex (biased toward the center). | Devroye (1986) §5.3 | No bug. If Dirichlet distribution is needed in future, add a separate `randomDirichletInpl` via Gamma sampling. |

---

## Summary of New Findings vs Round 1

1. **DingDong — ROUND 1 REFUTED**: The code is correct per NIST. Round 1's arithmetic slip in the index conversion produced a false Medium-severity finding. The Medium finding from round 1 should be retracted.

2. **Nearest-neighbor rounding (Low/NEW)**: `math.round` uses banker's rounding; MATLAB uses round-half-away-from-zero. Tie points only. Suggest a doc note or replace with `floor(pos+0.5)` if MATLAB parity is required.

3. **Cauchy convention (Info/NEW)**: Code uses `1/(x+y)` (MATLAB convention), not Wikipedia's `1/(x−y)`. The Hilbert-as-special-case example in the docstring confirms this is intentional. Suggest a one-line doc note for cross-reference clarity.

4. **Mertens values verified (Info)**: M(1..8) = {1,0,−1,−1,−2,−1,−2,−2} computed from first principles. Code comment is correct.

5. **Gallery closed-forms fully derived**: All 20+ matrices re-derived against primary sources (Smith's theorem for GCD, Parter 1986 for singular-value clustering, Mezzadri 2007 for Haar, etc.). No additional errors found beyond what round 1 already cleared.

6. **All ICDF samplers re-verified** with domain analysis (Weibull at u=0, Triangular point-mass guard, Pareto range). All correct.

7. **k-means++ D² incremental update verified** against Arthur & Vassilvitskii (2007). The GEMM identity for assignment and the final-sync invariant are both correct.
