# DEVLOG — TemplateSourceBenchmarks
Code comments state contracts only; history lives here (see CLAUDE.md).

## DetMathBenchmark
- 2026-07-15 (r7) | REMOVED the 0-ULP FLOAT CR variants (DetMathCR.cs) — user's call: float-CR
  without a double-CR counterpart is an asymmetry that feels wrong, and 1-ULP deterministic is the
  right target anyway (0-ULP adds cross-LIBRARY agreement + exact monotonicity, NOT cross-arch
  reproducibility, which 1 ULP already gives; and any downstream op's own rounding swamps a 1-ULP
  input error — log-det's sum accumulation is the example). The proof stands in history (commit
  b451441): float exp/log via double-eval are correctly-rounded over all 2^32 floats. If ever needed,
  double 0-ULP is reachable via double-double + Ziv rounding test + bounded triple-double fallback
  (deterministic, ~3-8× slower) — not built. Keeping the fast 1-ULP deterministic set as the design.
- 2026-07-15 (r6) | Added CORRECTLY-ROUNDED (0-ULP) FLOAT exp/log (user asked, for log-det etc.) —
  `Benchmarks/DetMathCR.cs` (float-only, not templated). Technique: evaluate the deterministic poly in
  DOUBLE (deg-11 exp / deg-8 B log, ~2^-56 rel — far below float's 2^-24 grid), round once to float.
  EXHAUSTIVELY PROVEN: a Burst job walks all 2^32 float bit patterns vs the double-libm reference
  rounded to float — logCR 2,139,095,039 floats (every positive finite) maxULP 0 / 0 mismatches;
  expCR 2,237,661,186 (every finite |x|<=88) maxULP 0 / 0 mismatches. So our float CR == libm-double-
  to-float everywhere = correctly rounded. Throughput (10M): logCR 44.4ms (still 1.8× < math.log 80.5,
  AND 0-ULP + deterministic); expCR 37.1ms (~on par with math.exp 33.6 — double eval costs ~2×, but
  deterministic + CR which libm is not). KEY POINTS: (1) float 0-ULP is EASY and provable (double
  headroom + only 2^32 inputs to check exhaustively). (2) DOUBLE 0-ULP is much harder — needs double-
  double (Dekker) intermediate precision, no exhaustive check possible (2^64) — a research-grade
  effort; NOT done. For log-det specifically the SUM accumulation error dominates the per-term ULP, so
  a ~1-2 ULP DETERMINISTIC double log already gives a reproducible accurate log-det; per-element double
  0-ULP rarely changes the sum. (3) GIVENS/JACOBI rotations need NOTHING from DetMath — the library
  computes (c,s)=(f/r,h/r), r=pythag=sqrt(f²+h²): sqrt+div only, both IEEE correctly-rounded &
  deterministic already (verified SVD.fProxy.cs / Eigen.fProxy.cs use jacobiRotate + pythag, no trig).
- 2026-07-15 (r5) | Added deterministic atan — the "one hard primitive", now DONE. A single wide-range
  poly is hopeless (deg-12 on [0,1] = 8e4 ULP double), so use libm-style reduction: fold |x|>1 via
  atan(x)=π/2-atan(1/x) → [0,1], then split at tan(π/8) via atan(x)=π/4+atan((x-1)/(x+1)) so the poly
  argument stays in ~[-0.29,0.41]; odd minimax atan(xr)=xr·P(xr²), float deg-4 double deg-10, all
  branch-free via math.select (both branches computed then blended → vectorizes; single π/4,π/2
  constants, no hi/lo split needed). BIGGEST speedup of the whole sweep because math.atan is the
  slowest transcendental: float math 156 → det 29.0 (5.4×) @1.33 ULP; double 171 → 47.9 (3.6×) @1.00.
  single float 303→112 (2.7×), double 308→181 (1.7×). **ALL 5 CORE PRIMITIVES NOW PROTOTYPED
  (batch speedup vs Unity.Mathematics, float / double, deterministic +-*/ & int, ~1-2 ULP):
  exp 2.2×/1.9× · sin 4.35×/2.7× · cos 4.45×/3.5× · log 4.6×/3.1× · atan 5.4×/3.6×.** Every core
  transcendental is 2-5.4× FASTER than libm on batch. DetMath is fully de-risked: feasible, faster,
  deterministic. Remaining before shipping: edge-case hardening (inf/NaN/subnormal/domain), Payne-
  Hanek for large trig args, Estrin as canonical eval, promote to TemplateSource/OP/DetMath.fProxy.cs
  + golden-vector tests per spec-detmath.md.
- 2026-07-15 (r4) | Added deterministic log (x=m·2^e via exponent bits, m centered to [√2/2,√2),
  log(m)=2s·B(s²) atanh form, s=(m-1)/(m+1); B minimax float deg-3 double deg-7). batch ms (10M):
  float math.log 81.6 → det 17.7 (4.6×) @1.35 ULP abs; double 94.5 → 30.4 (3.1×) @2.0. single float
  152→122 (1.25×); double single det 166 vs math 159 (Horner deg-7 chain latency — Estrin would fix,
  same pattern as exp double). **CONSOLIDATED across all prototyped fns (batch throughput speedup vs
  Unity.Mathematics, float / double, all deterministic +-*/ & int only, ~1-2 ULP):** exp 2.2× / 1.9×
  (Estrin), sin 4.35× / 2.7×, cos 4.45× / 3.5×, log 4.6× / 3.1×. So in-house deterministic DetMath is
  2-4.6× FASTER than libm on batch — the determinism (cross-arch bit-identical) is a perf WIN, not a
  tax, because libm hardens for full range + all edge cases while our bounded-range poly is lean.
  Trig/log reductions are bounded-arg (|x|<~10, sane mantissa); full range needs Payne-Hanek (trig) /
  edge handling (all). atan still deferred. Decision input: option-3 DetMath is strongly favored,
  default-on for Comp.*/softmax/dft. Estrin should be the canonical eval scheme (fixes double single
  latency across exp/log).
- 2026-07-15 (r3) | Added deterministic sin/cos (Cody-Waite π/2 reduction + odd/even minimax
  sin=r·P(r²) cos=Q(r²) + branch-free quadrant select; Remez coeffs via mpmath). **BIGGEST WIN so
  far — far bigger than exp, because math.sin/cos are much heavier (full reduction + poly) while our
  bounded-range version is lean.** 10M, Ryzen 9950X3D, batch ms: float sin math 108.9 → det 25.0
  (4.35×) @1.17 ULP; float cos math 116.5 → det 26.2 (4.45×) @1.13; double sin 91.3 → 33.6 (2.7×)
  @1.00; double cos 117.5 → 33.4 (3.5×) @1.00. single/latency float sin 218 → 115 (1.9×). Accuracy
  is absolute error (sin/cos hit zeros so relative blows up): float ~1.2 ULP, double ~1.0 ULP.
  Both auto-vectorized incl. the branch-free quadrant int ops. Consumers that benefit most: FFT dft
  twiddles, easing, Wave, rotations, windows/kernels — all bounded-argument. CAVEAT: reduction is
  2-part Cody-Waite → accurate for |x|<~10 only; large |x| needs Payne-Hanek (documented domain, same
  posture as exp). atan DEFERRED — needs interval-splitting reduction (deg-12 minimax on [0,1] still
  8e4 ULP double; libm reduces to smaller intervals). Coefficient sets recorded in the coder's remez
  scripts; regenerate via mpmath dps=50-60 Remez exchange.
- 2026-07-15 (r2) | Added MINIMAX coefficients + an ESTRIN-scheme variant (user asked). Genuine Remez
  minimax computed offline (mpmath dps=50, scaled-variable exchange): float deg-6 ~0.03 ULP fit,
  double deg-11 ~0.03 ULP fit (1 term fewer than the deg-12 Taylor). KEY LEARNINGS: (1) for exp,
  minimax ≈ Taylor — coefficients differ only in the 5th-7th sig fig (exp is entire/super-smooth; the
  minimax edge grows for functions with nearby singularities like log/atan, NOT exp). So minimax buys
  ~1 degree, not a revolution. (2) The double accuracy problem (1.42 ULP in r1) was the ln2 SPLIT, not
  the poly: switching to the fdlibm zero-low-bit hi/lo (0x3FE62E42FEE00000) dropped Horner double to
  1.00 ULP. (3) ESTRIN is the real latency win, exactly as the spec said to use it: float single
  108→72 ms (-33%), double single 160→89 ms (-45%). Horner's deg-11 chain (double single 160ms) is
  even SLOWER than math.exp single (139ms) — a long dependency chain loses to libm on per-call latency;
  Estrin's balanced tree fixes it. Estrin also helped batch (float 17.7→15.2). Numbers (10M, Ryzen
  9950X3D, ms): float batch math.exp 33.6 | det.acc Horner 17.7 @0.93ULP | det.acc Estrin 15.2 @1.18
  | fast 10.9 @8e-4. float single math.exp 148 | Horner 108 | Estrin 72 | fast 77. double batch
  math.exp 41.6 | Horner 33.0 @1.00ULP | Estrin 22.4 @1.94 | fast 17.8 @1.6e-7. double single math.exp
  139 | Horner 161 | Estrin 89 | fast 108. So best deterministic exp beats math.exp: float 2.2× batch
  / 2.05× single, double 1.86× batch / 1.57× single (both via Estrin). Estrin/Horner give different
  bits (different rounding order, both deterministic) → a shipping DetMath must pick ONE canonical.
- 2026-07-15 | New benchmark: native math.* vs a PROTOTYPE in-house deterministic exp (DetMathProto,
  benchmark-only, non-shipping — Cody-Waite reduction + poly + ldexp via exponent bits, only +-*/ and
  int/bit ops → cross-arch bit-identical by construction). Ryzen 9 9950X3D, 10M elements, batch =
  independent (vectorizable) vs single = dependent chain (per-call latency). HEADLINE, contra the
  "we won't beat libm" prior: the deterministic accurate poly BEATS math.exp on throughput at equal
  accuracy — float batch det.acc 21.3ms @0.68 ULP vs math.exp 34.8ms @0.66 ULP (1.6x); double batch
  det.acc 38.0ms @1.4 ULP vs math.exp 42.6ms @1.0 ULP. Fast tier: float 10.5ms @8e-4, double 16.3ms
  @1.6e-7. Both paths auto-vectorize (batch ~5.6x faster than single per elem → Burst vectorized our
  int-heavy reduction). Single/latency: det.acc float 120ms vs math.exp 152ms. So determinism here is
  free-to-profitable on throughput, because our poly is leaner than libm's fully-hardened path.
  CAVEATS before promoting to a shipping DetMath.Exp: (1) prototype skips overflow/underflow/NaN/inf
  edge handling (libm does it — costs some ops); (2) accuracy measured only on inputs [-10,10], not the
  full range / not guaranteed correctly-rounded everywhere; (3) double-acc uses a plain hi/lo ln2 split
  + Taylor deg-12 — a zero-low-bit Cody-Waite split + a minimax poly would BOTH tighten ULP and drop a
  term (faster); (4) double "ULP" is vs System.Math.Exp (same libm), so it's agreement, not true error.
  math.* float throughput baseline (10M): sin 108, cos 113, exp 34.6, log 82.8, atan 166.5 ms;
  double: sin 94, cos 118, exp 42.8, log 96, atan 186 ms.

## KMeansBenchmark
- 2026-07-12 | KMeansJobFProxy.Execute() called KMeans.fit with a hardcoded literal 16 while
  BenchFProxy sizes the centroids/workspace buffers from parameter K -- harmless only because the
  sole caller always passes K=16 (audit release-scan-2026-07-12/22-benchmarks finding 8, UNCERTAIN).
  Added a `K` field to the job and passed it through instead of the literal, so the requested cluster
  count and the buffer sizes are driven by one value. (was KMeansBenchmark.fProxy.cs:26)

## SparseSolverBenchmark
- 2026-07-12 | Dropped "Milestone B"/"Milestone-A" internal phase tags from three comments (the
  transpose-optimized CGLS/LSQR job banner, the Section 0b symmetric-vs-full spMV banner, and the
  Section-with-transpose-rows comment); kept the contracts (Aᵀ materialized once outside timing so
  ApplyT runs as a forward spMV; symmetric lower-triangle storage vs full block-CSR on the identical
  matrix). (was SparseSolverBenchmark.fProxy.cs:184, 647, 843)

## DirectSolveBenchmark
- 2026-07-12 | LuSolveTransAJobFProxy's comment dropped a perf-verdict/expectation clause ("this
  should run at roughly the same speed as the forward LU row -- any large gap would mean the
  right-looking TransA formulation isn't vectorising as intended"); kept the contract identifying
  the job as the getrs(trans='T') counterpart of LuSolveJobFProxy. (was
  DirectSolveBenchmark.fProxy.cs:39-41)

## LargeSparseBenchmark
- 2026-07-12 | Stripped remaining ticket tags and budget-bookkeeping narration missed by the previous
  pass (audit release-scan-2026-07-12/22-benchmarks finding 5): "Krylov R3"/"Krylov R3b" round labels
  on the SpCgJobFProxy/SpPcgJobFProxy tol-field comment, the SpPcgSSORJobFProxy comment, the LOBPCG
  Bench.Time wiring comment, and the SSOR-preconditioner-axis comment (also dropped its
  LobpcgAcceptsSSORPreconditioner test-name reference and "hypothesis under test" narration);
  BenchKrylovFProxy's comment dropped the "spMV x50 row DELETED... Q7 budget ruling" bookkeeping and
  the stale BenchPrecondConvergenceFProxy reference (that method was inlined, never renamed back in
  the comment); BenchStencilFProxy's comment dropped the "spec: BR=4/1.5% fill..." citation and the
  "gained PCG-SSOR... PAID FOR by dropping N=5120... Q7 budget ruling" history; BenchLobpcgFProxy's
  comment dropped the "Krylov R3b budget trade (spec §3b, disclosed)... DROPPED... pay for" narration.
  All now state only what each row set measures. (was LargeSparseBenchmark.fProxy.cs:25, 42, 73-81,
  90-97, 131-140, 238-242, 322-328)
- 2026-07-11 | Dropped all `docs/draft-spec-krylov-optimization.md` citations (round labels R3/R3b)
  across the file's header and section comments: the "iters+status makes breakdown-guard early-exits
  visible" hygiene note, the SpCgJobFProxy/SpPcgJobFProxy `tol`-field-reuse comment, the LOBPCG
  wall-clock-column comment, and the PCG preconditioner-axis section banner. Also trimmed the SSOR-for-
  LOBPCG hypothesis comment: dropped the "(R3, verified by ...)" and "(R3 finding)" round-number tags
  and a vague "(spec §3b/task brief)" aside, keeping the actual hypothesis (LOBPCG's per-iteration cost
  is Rayleigh-Ritz-dominated, so SSOR's iteration-count win might beat block-Jacobi's cheaper apply)
  and the fact that fProxySSOR routes through the generic `lobpcg<TOp,TPre>` core via
  fProxyBSROperator rather than a dedicated overload.

## LQRBenchmark
- 2026-07-12 | BuildInstanceFProxy's construction comment dropped its development history (the former
  fixed +-0.05 off-diagonal magnitude was only stable to n~12; at n=128 it produced unstable, likely
  unstabilizable instances; at n=4 the current 0.2/n scaling reproduces the original +-0.05 exactly).
  Kept the current contract (diagonal range, off-diagonal scaled 0.2/n, Gershgorin bound stays under
  ~0.6 at every n). (was LQRBenchmark.fProxy.cs:95-99)

## LPBenchmark
- 2026-07-12 | Trimmed three more comments to contract-only (audit release-scan-2026-07-12/22-benchmarks
  findings 1-3): the LadJobFProxy residual-recompute comment dropped its bug-postmortem clause (a
  not-quite-converged solve can report LPInfo.objective BELOW the true L1 residual, observed once as
  m=192 float revised printing 4.37 vs a true residual of 104.08, silently misleading the table); the
  LpRhsMatVecJobFProxy comment dropped the "coordinator's sanity-scan" workflow reference and the
  73728-multiply-add Mono-interpreted rationale; the SectionInfeasibleFProxy comment dropped the
  "(the same robust recipe the review that requested this section specified)" reviewer-provenance
  parenthetical. (was LPBenchmark.fProxy.cs:59-65, 174-179, 608-610)
- 2026-07-11 | Trimmed the killed-run anecdote from the per-job reporting-outputs comment. Full story:
  the report used to harvest objective/iters/status via a SEPARATE plain managed call to LP.solve/LP.lad
  before ever timing the Burst job -- i.e. every row solved the SAME problem TWICE, once fully
  Mono-interpreted. That was fine at n=24 but catastrophic at n=384 (seconds per solve): an extended
  benchmark run measured minutes and was killed because of it. The fix moved reporting into the timed
  job itself (objOut/itersOut/statusOut, length-1 arrays, written from inside Execute()), so
  Bench.Time's existing warmup + 4 timed .Run() calls populate them as a side effect of the SAME
  Burst-native call already being timed -- no second solve. Kept in-source: the current contract
  (self-reporting from inside Execute(), no second solve) and the enum-to-int cast note for
  LPBenchmarkFmt.InfeasRow. Also dropped dangling `docs/spec-revised-simplex.md stages 1+2` (Section 1
  banner) and `docs/spec-lpbasis-persistence.md acceptance item 5` (Section 6 banner) citations.
  (was LPBenchmark.fProxy.cs:24-35, 253, 310)

## KalmanBenchmark
- 2026-07-12 | Section 4's drift-safety comment (EkfCycleFProxy) was arithmetically wrong and
  cited the wrong regime: it claimed NonlinearSteps kept the cumulative step count "well inside"
  KalmanTests.fProxy.cs's 80-step EKF acceptance test, but NonlinearSteps=100 already exceeds 80
  per Execute() and Bench.Time's 1 warmup + 4 timed calls on the same persistent state push the
  true cumulative to ~500 -- and that acceptance test tracks a real noisy trajectory with seeded
  P, not this zero-measurement self-simulation, so it never validated this regime anyway. Rewrote
  the comment with the actual bound (~500 cumulative) and the real reason boundedness holds here:
  Smeas = H P Hᵀ + R stays positive as long as R > 0, independent of how far the self-simulated
  state/covariance drift, so the Cholesky-based innovation solve does not trip
  InnovationSolveFailed on this model. Noted it applies identically to UkfCycleFProxy, which had
  no drift note at all.
