# W3 - Numerical correctness report

Scan of templates under Assets/LinearAlgebra/CodeGen/TemplateSource* for
numerical-safety issues (division/normalization by zero, exact float equality,
per-type tolerance drift, integer overflow, symmetry loss, convergence-check
scaling).

Overall the codebase is disciplined: most tolerances are wired through
Consts.fProxy* (which resolves to per-type float* / double* values via the
fProxy -> float/double filename split), pivots are guarded with NaN-safe
!(x > 0) patterns almost everywhere they matter (CHO, LOBPCG.FactorGram,
Krylov breakdowns, LSQR/LSMR rotations), sqrt inputs are non-negative by
construction (dot(v,v), squared norms), and division-by-zero is either guarded
or documented (Blas triangular solves).

Findings below are ordered highest severity first.

---

## HIGH findings

None. No unconditional division-by-zero, no unguarded sqrt of a possibly-
negative value, no lost-symmetry bug, no per-type constant that would silently
mis-fire in the double variant of a shipped hot path.

---

## MEDIUM findings

### M1 - absInPlace on iProxy silently returns MinValue for MinValue inputs; not documented on the public API

Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeMathOP.iProxy.cs:41-47

    public static void abs([NoAlias] iProxy* x, int n)
    {
        for (int i = 0; i < n; i++) {
            iProxy v = x[i];
            x[i] = v < 0? (iProxy)(-v) : v;
        }
    }

Twos-complement wraps for -int.MinValue, -long.MinValue, and (via short-cast
truncation of the promoted int) -short.MinValue. In every one of those cases
the absolute value written back is the SAME negative MinValue, silently
producing a negative element where the caller expects a non-negative one.

NormsOP.iProxy.cs documents this trap for L1/LInf (only for the long variant,
because L1/LInf widen to long inside math.abs((long)a.Data[i]) first, which
fixes int/short). The absInPlace kernel is called from
OP.Component.iProxy.cs:274

    public static void absInPlace<T>(this T x) ...
    {
        unsafe { UnsafeMathOP.abs(x.Data.Ptr, x.Data.Length); }
    }

whose XML doc says nothing about this behaviour, and the same trap fires for
EVERY generated signed type (int/short/long), not just long. Concrete failing
scenario: a caller does intVec.absInPlace() on a vector containing
int.MinValue and gets int.MinValue back where they expect 2^31 (which cannot
be represented) - the docs give them no warning that this is possible.

Fix direction: add the wraps-for-MinValue caveat to absInPlace XML doc and
to UnsafeMathOP.iProxy.cs abs doc; do not attempt to widen inside the kernel
(matches the documented library-wide signed integer MinValue wraps
convention).

---

## LOW findings

### L1 - Debug.Spy(m) default threshold 0.01f loses precision when generated for double

Assets/LinearAlgebra/CodeGen/TemplateSource/Debug/Debug.fProxy.cs:88

    public static void Spy(in fProxyMxN m) => Spy(m, 0.01f);

The 0.01f is a float literal. In the generated double variant this becomes
Spy(in doubleMxN m) => Spy(m, 0.01f); where the float 0.01f is widened to
double (0.009999999776482582...), not the double 0.01. Consequence is
invisible here (a sparsity-plot threshold), but the same pattern in a hot
numerical kernel would round its default tolerance to float precision.

Fix direction: use (fProxy)0.01 (unsuffixed literal), or route through a
Consts.fProxy* per-type value.

### L2 - Kalman.numericJacobianF/H finite-difference step is not scaled by |x[k]|

Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Kalman.fProxy.cs:413-437 (F)
and :449-473 (H). The step is a plain constant eps:

    xp[k] += eps;
    xm[k] -= eps;
    ...
    J[i, k] = (fp[i] - fm[i]) / ((fProxy)2 * eps);

For a state component with |x[k]| >> 1 the finite-difference error grows
linearly in |x[k]| because the argument perturbation is unchanged, so the
Jacobian gets progressively noisier as the state grows. NLS.fProxy.cs gets
this right at line 112: fProxy step = hEps * math.max(math.abs(pj),
(fProxy)1); (the standard MINPACK convention).

Not a HIGH because the doc gives eps to the caller and the eps-default
overload uses Consts.fProxySqrtEps (a reasonable scale for O(1) states).
Would misbehave silently on a body-frame position filter or any state with
large magnitude.

Fix direction: match NLS scaling (hEps * max(|x[k]|, 1)) inside the Kalman
numeric-Jacobian helpers, or document that eps is expected to already carry
the state scale.

### L3 - BSR block-CSR kernels use int multiplication for pointer offsets; overflow-unsafe for very large sparse matrices

Assets/LinearAlgebra/CodeGen/TemplateSource/Sparse/UnsafeOP.Sparse.fProxy.cs
throughout the general and B2/B3/B4/B6 kernels, e.g. lines 25 (int yBase =
br * BR;), 30 (int xBase = bc * BC;), 31 (values + k * blockLen), 106 (int
yBaseI = bi * BR;), 112 (values + k * blockLen), 187, 188, 210, 211, etc.
The dense factorization kernels consistently use (long)i * m casts (see
LU.fProxy.cs, CHO.fProxy.cs); the BSR kernels do not, which caps their safe
pointer arithmetic at ~2^31 elements.

Concrete failing scenario: a matrix with 1e8 stored blocks and blockLen=64
(8x8 blocks) has k * blockLen = 6.4e9, wrapping int and producing garbage
pointer arithmetic.

Fix direction: cast to long in the same places the dense kernels do -
values + (long)k * blockLen, br * (long)BR, etc. - across all B1/B2/B3/B4/B6
and general/symmetric variants.

### L4 - Control.SymmetrizeInPlace sum-then-halve overflows for near-MaxValue entries

Assets/LinearAlgebra/CodeGen/TemplateSource/OP/Control.fProxy.cs:108

    fProxy avg = (M[i, j] + M[j, i]) / (fProxy)2;

If M[i,j] and M[j,i] are both near fProxy.MaxValue the sum overflows to +Inf
before the divide. Practical release-blocker chance ~0 (this runs inside the
LQR/DARE hygiene loop, which would already be Diverged at that scale), but
the safe form M[i,j] * 0.5f + M[j,i] * 0.5f costs nothing.

Fix direction: half-then-sum ordering as above.

### L5 - iProxyN.Operators.cs / iProxyMxN.Operators.cs divide-by-zero guards compare integer against 0f

Assets/LinearAlgebra/CodeGen/TemplateSource/iProxy/iProxyN.Operators.cs:78,
:101; iProxy/iProxyMxN.Operators.cs:76, :96:

    if (s == 0f)
        throw new DivideByZeroException();

s is iProxy (int/short/long/uint after codegen). The comparison relies on
the implicit iProxy -> int -> float widening; 0 is exactly representable in
float, so the guard works correctly, but the FORM is misleading (looks like
a float check) and inconsistent with every other divide-guard in the library
(s == 0 or s == (iProxy)0).

Fix direction: replace 0f with 0 (or (iProxy)0) - a purely stylistic fix,
no behaviour change.

### L6 - LP.DualSimplex row-perturbation base 1e-12 has no effect for the float variant

Assets/LinearAlgebra/CodeGen/TemplateSource/OP/LP.DualSimplex.fProxy.cs:334

    double rowPerturbBase = 1e-12;
    ...
    perturbedCost[j] = cost[j] + (fProxy)((0.5 - r) * rowPerturbBase);

For fProxy == float the perturbation magnitude is ~5e-13, well below float
~1e-7 relative epsilon at O(1) cost scales, so cost[j] + (float)5e-13 rounds
back to cost[j] exactly - the row (logical column) tie-break is a no-op in
the float variant. The comment on the following line explicitly claims
Bases are HiGHS own literals for BOTH dtypes (both representable in float)
which is true of the base value in isolation but not of the ADDITION at the
intended precision. LOW because the outcome is no tie-break not wrong
tie-break, so simplex just relies on other tiebreakers.

Fix direction: either scale the row base by Consts.fProxyEpsilon (per-type
epsilon, ~7 orders below 1e-7 for float), or note in the DEVLOG that the
row perturbation is intentionally double-only.

### L7 - SVD Wilkinson-shift denominator can be zero when d[k-1] is degenerate after a cancellation sweep

Assets/LinearAlgebra/CodeGen/TemplateSource/OP/SVD.fProxy.cs:462 and :641

    fProxy f2 = ((yy - zz) * (yy + zz) + (g2 - h2) * (g2 + h2)) / ((fProxy)2 * h2 * yy);

yy = d[k-1]. The upstream deflation guard only rejects |e[l]| <= thresh
before computing the shift; a small |d[k-1]| breaks out of the flag loop
into the CANCELLATION branch which then falls through to this Wilkinson
compute with yy still ~0. Followed a few lines later by another divide by
x = d[l] on the same expression. Very well-known code from Numerical Recipes
svdcmp, and empirically robust because the cancellation branch usually
rescales things before flow reaches here - but there is no explicit guard
that would prevent Inf/NaN in an adversarially-constructed input.

Fix direction: document the assumption on the pre-loop invariant, or add a
math.max(|yy|, tiny) floor consistent with the thresh-based deflation. Not
a blocker; this is the shipped-for-decades LAPACK/NR pattern.

### L8 - iProxy dot / sum accumulators overflow silently in the same type

Assets/LinearAlgebra/CodeGen/TemplateSource/OP/UnsafeMathOP.iProxy.cs:96-104

    public static iProxy dot([NoAlias] iProxy* x, [NoAlias] iProxy* y, int n)
    {
        iProxy sum = 0;
        for (int i = 0; i < n; i++)
            sum += (iProxy)(x[i] * y[i]);
        return sum;
    }

For int inputs each product x[i] * y[i] can already overflow (e.g. x[i] =
y[i] = 2^16 gives 2^32 which wraps int), and the accumulator wraps again.
Contrast: NormsOP.iProxy.cs::L1 widens to long before summing, L2 widens to
double, StatsCore.iProxy.cs::sum widens to long. Blas.dot for iProxy uses
this same in-type accumulator and returns iProxy (int/short/long/uint), so a
caller of Blas.dot(intVec, intVec) on non-tiny inputs silently gets a
wrapped result.

LOW rather than MEDIUM because (a) the return TYPE is iProxy so widening
would break the API shape, and (b) the fProxy version is the real dot that
people actually use in numeric code. Integer dot is documented as integer-
shape, so this is arguably by contract - but the contract itself is sharp
and worth adding to the XML doc.

Fix direction: add overflow-risk caveats to Blas.dot(iProxyN, iProxyN) XML
doc (OP.Dot.iProxy.cs:17-25), matching the same caveats sum/mean already
carry in StatsCore.iProxy.cs.

---

## Areas confirmed clean

- Cholesky (OP/CHO.fProxy.cs): NaN-safe pivot check !(d > 0) before sqrt on
  every path (unblocked, blocked, in-place). Triangular back-solve is the
  only place that unconditionally divides by L[r,r], and this is documented
  as a caller precondition consistent with Blas.triLower / triUpper.
- Pivoted Cholesky (OP/CHOP.fProxy.cs): scale-relative stopTol = n * eps *
  absScale (scale-invariant), catches NaN via !(maxDiag > stopTol), detects
  Indefinite via minDiag < -stopTol AND via the trailing-mass check, no
  unguarded sqrt.
- LU (OP/LU.fProxy.cs): exact-zero pivot check for decompNoPivot (documented
  contract), max-abs pivot search + pivotValue == 0 (post-abs) check for the
  partial-pivoted variants, both unblocked and blocked. Final-diagonal check
  after the k<m-1 loop.
- QR/QRCP/LQ (OP/QR.fProxy.cs, OP/QRCP.fProxy.cs, OP/LQ.fProxy.cs): all use
  Consts.fProxyZeroThreshold * Norms.LInf(A) for the scale-relative zero
  column detection. Householder sign convention (signOrOne(0) == +1) is
  documented in OP/OpHelpers.fProxy.cs:14-19 and matches Fortran SIGN.
- Bidiag / SVD (OP/SVD.fProxy.cs): pythag-guarded sqrt of f*f+g*g throughout,
  deflation threshold scaled by global anorm (not local), values-only path
  matches the full path scalar-for-scalar. One residual concern in L7 above
  (Wilkinson shift denominator) is the NR/LAPACK shipped pattern.
- Symmetric eigen QL / Householder tridiagonalization (OP/Eigen.fProxy.cs):
  NR copysign convention, gammaFloor-ed Givens denominator (matches MINRES
  pattern).
- MINRES / CG / PCG / BiCGStab / CGLS / LSQR / LSMR / CGNE
  (OP/Krylov.fProxy.cs): every division-by-scalar-norm is either preceded by
  an !(norm > 0) breakdown check, or the norm is a squared dot that has
  already been > threshold tested. bb == 0 shortcut is NaN-sanitising.
  Verify-at-exit path recomputes r fresh on the CG family to catch drift-to-
  false-convergence in float.
- LOBPCG (OP/LOBPCG.fProxy.cs): FactorGram has Tikhonov-ridge retry with
  scale-relative shift, MinMaxDiagRatio catches numerical rank deficiency,
  envelope-based safeguard around the small eigenproblem rejects garbage
  Ritz values, bn2 > 0 guard on each B-orthonormalization renormalize.
- IC0 preconditioner (Sparse/fProxyIC0.cs): pivotFloor = 16 * eps * diagMax
  (scale-relative), NaN-safe !(sum > pivotFloor) reject, escalating-shift
  retry.
- Blas Jacobi scale (OP/Blas.ColumnScaling.fProxy.cs): NaN-safe c > 0
  fallback to d[j] = 1 for a zero column - no divide by zero.
- Iterative solve threshold pattern: everywhere I checked, the convergence
  test is squared-relative (rr <= tolerance * tolerance * bb), never
  absolute, so the tolerance interpretation is scale-invariant.
- LP simplexes and interior-point (OP/LP.*.fProxy.cs, OP/QP.fProxy.cs):
  per-type feasTol = max(sqrt(eps), 1e-7) and pivTol = max(Consts.
  fProxyZeroThreshold, 1e-9) inlined; scale-relative artificial bounds;
  float-vs-double tie tolerance handled via //+choose[1e-5f|1e-9],
  //+choose[1e-6f|1e-12] where it matters.
- NLS (OP/NLS.fProxy.cs): finite-difference step correctly scaled by
  max(|p_j|, 1). Robust-loss J-scale floored at Consts.fProxyEpsilon before
  the sqrt, avoiding the redescending-loss negative-sqrt trap. Scale vector
  d is proven never-zero on the used path.
- Kalman KF / EKF / UKF (OP/Kalman.*.fProxy.cs): Joseph-form P update (not
  the classic (I-KH)P float divergence source), Wc[0]<0 risk is explicitly
  called out and symmetrized. K solved via CHOP transposed system on both KF
  and UKF, never an explicit inverse. steadyStateGain rescales Q/R to unit-
  norm before SDA so the LQR-tuned convergence floor applies.
- MPC (OP/MPC.fProxy.cs): warm-start plan is feasible-by-construction
  (clipped, LQR tail-filled, forward-simulated). No numerical decision uses
  an unguarded division. Fallback path is captured BEFORE the QP mutates
  state.
- Statistics (Statistics/StatsCore.iProxy.cs): long-widened sum accumulator,
  double variance/std, all documented long.MinValue limitations pinned. No
  numerical bugs in the float variant either.
- BSR spMV specializations (Sparse/UnsafeOP.Sparse.fProxy.cs
  B1/B2/B3/B4/B6): accumulation order is documented as bit-identical to the
  general kernel; spot-checked B2/B3/B4 and confirmed.
- Codegen splits: every place I checked with a per-type numeric constant
  either routes through Consts.fProxy* (which resolves to Consts.float* /
  Consts.double*), uses a //+choose inline marker, or forms the constant
  from Consts.fProxyEpsilon derivatives. No leaked float.Epsilon / MathF.*
  calls found in fProxy templates.

---

## Summary table

| Severity | Count |
|---|---|
| HIGH | 0 |
| MEDIUM | 1 |
| LOW | 8 |
