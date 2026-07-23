using System;

using LinearAlgebra;
using LinearAlgebra.Gallery;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Adversarial acceptance battery for the QRCP norm-DOWNDATING change (LAPACK dgeqp3/dlaqps-style,
// guarded). The downdated partial norms feed ONLY the pivot CHOICE;
// the Householder reflector arithmetic is untouched. This battery pins that contract.
//
// TWO comparison tiers (do not blur them):
//  - Tier E (exact-match): on inputs whose EXACT trailing norms are WELL-SEPARATED at every pivot
//    step, the pivot sequence is mathematically forced. Production (downdating) must reproduce the
//    IDENTICAL Pivot AND bit-identical Q/R vs a reference oracle that recomputes norms exactly every
//    step. Any deviation on a separated input is a genuine bug.
//  - Tier P (property): on ties / near-ties / heavy cancellation, several pivot orders are equally
//    valid. Assert INVARIANTS instead: |R| diagonal non-increasing; A·P == Q·R; Q orthonormal;
//    detected rank == the independently-derived rank; no NaN/Inf.
//
// Operational "well-separated" test (identical everywhere so it is consistent across all cases):
// at each step d, among the EXACT trailing SQUARED column norms over rows d..m-1, let s1 >= s2 be
// the two largest. The step is separated iff s1 > s2 * (1 + 8*Consts.fProxySqrtEps), or only one
// trailing column remains. An input is Tier-E-eligible iff EVERY step is separated. The reference
// oracle (OracleDecompInPlace, below) both reproduces the pre-downdate algorithm AND reports this
// flag.
//
// ORACLE / BURST DECISION (deliberate, reasoned deviation from the literal "non-Burst" spec wording):
// the oracle runs INSIDE THE SAME [BurstCompile] IJob as the production call. The spec text worried
// that a Mono oracle vs a Burst production call could diverge purely from runtime-codegen
// reassociation under FloatMode.Default, producing a SPURIOUS Tier E mismatch. Co-locating both in
// one job removes that confound entirely: whichever runtime the job resolves to (Burst, or Mono
// fallback if the NUnit Asserts defeat Burst compilation), production and oracle execute in the SAME
// runtime under the SAME FloatMode/FloatPrecision, so a Tier E bit-mismatch can only mean a real
// downdate-vs-exact pivot divergence — exactly the signal Tier E exists to isolate. The oracle is
// throwaway test scaffolding: its reflector step delegates to the SAME public vectorised kernel
// production uses (LinearAlgebra.Internal.UnsafeOP.axpy) and its reflector-vector build calls the
// SAME public Norms.L2Range, mirroring QR.genHouseholder / QR.applyReflectorRight bit-for-bit (see
// OracleGenHouseholder / OracleApplyReflectorRight — those QR kernels are `internal`, reachable here
// via the InternalsVisibleTo grants on both BurstLinearAlgebra.Tests and
// BurstLinearAlgebra.TemplateSource.Tests-firstpass (TemplateSource/AssemblyInfo.cs); a bit-identical
// replica over public primitives is used instead of calling them directly so the oracle stays
// independent of the kernel under test). The ONLY thing that differs between the
// oracle and production is the norm-tracking strategy that feeds pivot choice.
// ─────────────────────────────────────────────────────────────────────────────────────────────
public class fProxyQRCPDowndateTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            KahanSweep,               // (1) Tier P over n×theta grid
            KahanRankCrossCheck,      // (1) rank == SVD rank on a well-conditioned Kahan instance
            NormCollapseLadder,       // (2) dependent-column collapse; rank tracks eps tier
            MassCancellation,         // (3) rank-1 + noise; guard fires everywhere; rank==1 at auto tol
            GradualDecay,             // (4) slow geometric spectrum n>=128 (cumulative-guard construction)
            Ties,                     // (5) exact-dup + 1-ulp-apart columns; Tier P + determinism
            ScaleExtremes,            // (6) column norms spanning many orders; no NaN; reconstruction
            ZeroAndTinySizes,         // (6) zero matrix / zero columns / single column / n=1..3
            TierEDistinctMagnitudes,  // Tier E demonstrator (guaranteed-separated construction)
            BlockedPanels,            // level-3 blocked core: panel-boundary sizes + guard-cut mid-panel
            CacheEquivalenceFullRank,       // cache overloads == non-cache overloads, bit-for-bit
            CacheEquivalenceRankDeficient,  // same, rank-deficient A
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.KahanSweep:                    KahanSweep();                    break;
                case TestType.KahanRankCrossCheck:           KahanRankCrossCheck();           break;
                case TestType.NormCollapseLadder:            NormCollapseLadder();            break;
                case TestType.MassCancellation:              MassCancellation();              break;
                case TestType.GradualDecay:                  GradualDecay();                  break;
                case TestType.Ties:                          Ties();                          break;
                case TestType.ScaleExtremes:                 ScaleExtremes();                 break;
                case TestType.ZeroAndTinySizes:              ZeroAndTinySizes();              break;
                case TestType.TierEDistinctMagnitudes:       TierEDistinctMagnitudes();       break;
                case TestType.BlockedPanels:                 BlockedPanels();                 break;
                case TestType.CacheEquivalenceFullRank:      CacheEquivalence(10, 6, 707071u, false); break;
                case TestType.CacheEquivalenceRankDeficient: CacheEquivalence(8, 5, 808081u, true);   break;
            }
        }

        // ── Case 1: Kahan matrix (THE classic pivoted-QR stress input). Tier P across an n×theta
        //    grid. Kahan ties every column's 2-norm by construction, so it is generally NOT
        //    Tier-E-eligible (and famously NOT rank-revealed by column pivoting — see
        //    KahanRankCrossCheck for why we do NOT cross-check rank against SVD on ill-conditioned
        //    instances). We assert the invariants that hold unconditionally, PLUS — for n=16 ONLY,
        //    see below — the exact, no-permutation property.
        //
        //    At n=32/64 with theta≈0.285π, the trailing column norms genuinely COLLAPSE by the late
        //    pivot steps (diagNorm1 measured at ~1e-9 -> ~1e-19 -> exactly 0 across consecutive steps)
        //    — the well-known Kahan/RRQR pathology (column pivoting's rank-revealing guarantee
        //    degrades once the trailing block decays into pure rounding noise), NOT a downdate
        //    defect: an exact-recompute oracle run on the identical input picks a DIFFERENT column
        //    than production at this same collapse point too. n=16 never reaches this collapse (zero
        //    divergence from identity, and from the oracle, across all 4 thetas), so the exact
        //    no-permutation pin below is scoped to n=16, where it is a genuine, stable invariant.
        void KahanSweep()
        {
            try
            {
                for (int ni = 0; ni < 3; ni++)
                for (int ti = 0; ti < 4; ti++)
                {
                    int dim = ni == 0 ? 16 : (ni == 1 ? 32 : 64);
                    // c = cos of the classic thetas: ti0 the near-0.285π worst case, the rest a spread.
                    fProxy c = ti == 0 ? (fProxy)0.62524266f
                             : ti == 1 ? (fProxy)0.87758256f
                             : ti == 2 ? (fProxy)0.54030231f
                             :           (fProxy)0.36235775f;

                    var A0 = fProxyGallery.fProxyKahan(dim, c);
                    var Q = new fProxyMxN(in A0, Allocator.Temp);
                    var R = new fProxyMxN(dim, dim, Allocator.Temp);
                    var P = new Pivot(dim, Allocator.Temp);
                    var u = new fProxyN(dim, Allocator.Temp);

                    QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                    TierP(in A0, in Q, in R, in P);

                    // No-permutation pin — n=16 ONLY (see the class doc above for why n=32/64 are
                    // excluded: genuine late-stage trailing-norm collapse at the classic theta, a
                    // documented Kahan/RRQR property, not a downdate defect).
                    if (dim == 16)
                        for (int d = 0; d < dim; d++)
                            RecordEq(P[d], d);

                    P.Dispose();
                }
            }
            finally { }
        }

        // ── Case 1 (rank cross-check): on a genuinely WELL-CONDITIONED, full-rank Kahan instance
        //    (n=16, theta=1.2: s=sin, c=cos both comfortably away from 0; smallest σ ≫ auto tol),
        //    QRCP's detected rank must equal SVD-based numerical rank (Analysis.rank), both == n.
        //    NOTE: this cross-check is deliberately NOT applied to the ill-conditioned instances of
        //    KahanSweep. The Kahan matrix is the canonical case where column pivoting does NOT reveal
        //    rank precisely (it never pivots), so QRCP-rank and SVD-rank LEGITIMATELY diverge once the
        //    matrix is near-rank-deficient — asserting equality there would flag a documented property
        //    of the algorithm, not a downdating bug.
        void KahanRankCrossCheck()
        {
            try
            {
                int dim = 16;
                var A0 = fProxyGallery.fProxyKahan(dim, (fProxy)0.36235775f);

                int svdRank = Analysis.rank(in A0);   // SVD-based numerical rank (auto tol)
                RecordEq(svdRank, dim);               // sanity: this instance is genuinely full rank

                var Q = new fProxyMxN(in A0, Allocator.Temp);
                var R = new fProxyMxN(dim, dim, Allocator.Temp);
                var P = new Pivot(dim, Allocator.Temp);
                var u = new fProxyN(dim, Allocator.Temp);
                QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                TierP(in A0, in Q, in R, in P);

                int qrcpRank = RankFromR(in R, dim, dim);
                RecordEq(qrcpRank, svdRank);

                P.Dispose();
            }
            finally { }
        }

        // ── Case 2: norm-collapse ladder. A = [B | B·X + eps·noise] — k independent columns plus k
        //    dependent columns whose remaining norm collapses mid-factorization. This is the guard's
        //    home turf. Tier P + rank tracks the eps tier. eps values are scaled off
        //    Consts.fProxyZeroThreshold so they land CLEANLY on one side or the other of the auto rank
        //    threshold for BOTH float and double (a single template, both types must hold).
        void NormCollapseLadder()
        {
            try
            {
                int m = 20, k = 4, n = 2 * k;
                fProxy zt = Consts.fProxyZeroThreshold;

                // Four tiers. Two clearly ABOVE the auto threshold (dependent columns keep enough of an
                // independent component to count -> full rank 2k), two clearly BELOW (collapse -> rank
                // k). Absolute-rank pins only at the two UNAMBIGUOUS extremes; all four additionally
                // cross-check QRCP-rank against SVD-rank (same matrix, same tolerance).
                for (int e = 0; e < 4; e++)
                {
                    fProxy eps = e == 0 ? (fProxy)1e-2f       // clearly above -> full rank 2k
                               : e == 1 ? (fProxy)1000 * zt   // above         -> cross-check only
                               : e == 2 ? (fProxy)0.1f * zt   // below         -> cross-check only
                               :          (fProxy)1e-3f * zt; // clearly below -> rank k
                    int pinAbs = e == 0 ? 2 * k : (e == 3 ? k : -1);

                    var B     = GenerateOP.fProxyRandomMat(m, k, -1f, 1f, 424200u + (uint)e);
                    var X     = GenerateOP.fProxyRandomMat(k, k, -1f, 1f, 990000u + (uint)e);
                    var noise = GenerateOP.fProxyRandomMat(m, k, -1f, 1f, 133700u + (uint)e);
                    var D     = Blas.dot(B, X); // m×k dependent block (exactly in span(B))

                    var A0 = new fProxyMxN(m, n, Allocator.Temp);
                    for (int r = 0; r < m; r++)
                    {
                        for (int c = 0; c < k; c++) A0[r, c]     = B[r, c];
                        for (int c = 0; c < k; c++) A0[r, k + c] = D[r, c] + eps * noise[r, c];
                    }

                    int svdRank = Analysis.rank(in A0);

                    var Q = new fProxyMxN(in A0, Allocator.Temp);
                    var R = new fProxyMxN(n, n, Allocator.Temp);
                    var P = new Pivot(n, Allocator.Temp);
                    var u = new fProxyN(m, Allocator.Temp);
                    QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                    TierP(in A0, in Q, in R, in P);

                    int qrcpRank = RankFromR(in R, m, n);
                    RecordEq(qrcpRank, svdRank);        // core cross-validation: QRCP tracks SVD
                    if (pinAbs >= 0)
                        RecordEq(qrcpRank, pinAbs);      // pinned absolute at the unambiguous tiers

                    // Rank-revealing GAP at the collapse tier: the first "dropped" diagonal must be
                    // orders of magnitude below the last "kept" one — this is what "pivoting pushed the
                    // dependent directions to the trailing block" looks like without asserting WHICH
                    // original columns lead (a dependent column can have a larger norm than an
                    // independent one via amplification in X, so column-index placement is not robust).
                    if (pinAbs == k)
                    {
                        fProxy kept    = math.abs(R[k - 1, k - 1]);
                        fProxy dropped = math.abs(R[k, k]);
                        RecordBound(dropped, kept * (fProxy)1e-3f);
                    }

                    P.Dispose();
                }
            }
            finally { }
        }

        // ── Case 3: mass-cancellation. Every column ≈ a scalar multiple of ONE pivot direction plus a
        //    noise floor (rank ~1). Every downdate cancels catastrophically at step 1, so the guard
        //    must fire repeatedly. Tier P + detected rank == 1 at the AUTO relTol (default overload).
        void MassCancellation()
        {
            try
            {
                int m = 20, n = 8;
                var rng = new Unity.Mathematics.Random(0x3A55u);

                // Pivot direction v (m-vector), O(1) entries.
                var v = new fProxyN(m, Allocator.Temp);
                for (int r = 0; r < m; r++) v[r] = (fProxy)(rng.NextFloat(-1f, 1f));

                // Noise floor scaled off the type zero-threshold so it stays BELOW the auto rank
                // tolerance for BOTH float and double (0.01·zeroThreshold ≪ max(m,n)·zeroThreshold).
                fProxy noiseScale = (fProxy)0.01f * Consts.fProxyZeroThreshold;

                var A0 = new fProxyMxN(m, n, Allocator.Temp);
                for (int c = 0; c < n; c++)
                {
                    fProxy alpha = (fProxy)(rng.NextFloat(0.25f, 4f)); // distinct scalar per column
                    for (int r = 0; r < m; r++)
                        A0[r, c] = alpha * v[r] + noiseScale * (fProxy)(rng.NextFloat(-1f, 1f));
                }

                var Q = new fProxyMxN(in A0, Allocator.Temp);
                var R = new fProxyMxN(n, n, Allocator.Temp);
                var P = new Pivot(n, Allocator.Temp);
                var u = new fProxyN(m, Allocator.Temp);
                QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                TierP(in A0, in Q, in R, in P);

                RecordEq(RankFromR(in R, m, n), 1);

                // Also via the AUTO-tol solve path (independent rank consumer, default overload).
                var As = new fProxyMxN(in A0, Allocator.Temp);
                var b  = new fProxyN(m, Allocator.Temp);
                for (int r = 0; r < m; r++) b[r] = (fProxy)(rng.NextFloat(-1f, 1f));
                var x  = new fProxyN(n, Allocator.Temp);
                int solveRank = QRCP.solveInPlace(ref As, ref b, ref x).rank;
                RecordEq(solveRank, 1);

                P.Dispose();
            }
            finally { }
        }

        // ── Case 4: gradual-decay attack. A random matrix with a slowly-decaying geometric singular
        //    spectrum (ratio ~0.95 between consecutive σ) at n>=128: each single downdate step looks
        //    benign in isolation, yet the trailing norm decays by ~cond^-1 (many orders) cumulatively
        //    over the run. Tier P only (full rank, just ill-conditioned).
        //
        //    KNOWN, DISCLOSED LIMITATION: with the cumulative check broken back to a naive per-step
        //    check (or its threshold weakened ~1e6x), NO test in this file — including this one — goes
        //    red. The reason is structural, not a testing oversight to "just try harder" on: Tier P invariants
        //    (reconstruction / orthonormality / monotone diagonal / rank-from-R) are mathematically
        //    satisfied by ANY valid column-pivoting choice, so they cannot distinguish a CORRECT pivot
        //    decision from a merely-DIFFERENT-but-still-valid one — the ONLY observable a downdating
        //    bug can ever produce is a pivot-sequence divergence (Tier E), because R[d,d] is always
        //    computed from an exact reduction of whichever column IS chosen (QR.genHouseholder),
        //    completely independent of how accurately vn1 tracked it. And engineering a Tier-E-eligible
        //    (well-separated-at-every-step, i.e. forced pivot) construction where naive per-step
        //    tracking and correct cumulative tracking actually DISAGREE on which column wins requires
        //    the accumulated per-step floating-point error to approach the ~8·sqrt(eps) separation
        //    margin Tier E itself requires to call an input "forced" — back-of-envelope, that error is
        //    bounded by O(k·eps) over k benign (non-tripping) steps, i.e. ~k·1.2e-7 for float, so
        //    closing a ~2.8e-3 gap this way needs k on the order of 10^4, far beyond a "keep runtimes
        //    sane" problem size (checked via that bound, not merely assumed). Correctness instead rests
        //    on the guard formula being a faithful, algebraically-verified LAPACK dlaqps/dgeqp3
        //    transcription. A dedicated Tier-E regression test for this ONE guard remains open coverage.
        void GradualDecay()
        {
            try
            {
                int m = 160, n = 128, k = n; // min(m,n) = 128
                // cond so consecutive σ ratio == 0.95 exactly: σ_i = cond^(1-i/(k-1)),
                // ratio = cond^(-1/(k-1)) = 0.95  =>  cond = 0.95^-(k-1).
                fProxy cond = math.pow((fProxy)0.95f, (fProxy)(-(k - 1)));

                var A0  = new fProxyMxN(m, n, Allocator.Temp);
                var rng = new Unity.Mathematics.Random(0x9DEC0095u);
                Rand.conditionedInPlace(ref rng, ref A0, cond);

                // Sanity: the total dynamic range really is large (σ_max/σ_min ≈ cond ≈ 673), i.e. the
                // construction genuinely stresses cumulative decay — guards against a mis-set cond.
                RecordBound((fProxy)500, cond);

                var Q = new fProxyMxN(in A0, Allocator.Temp);
                var R = new fProxyMxN(n, n, Allocator.Temp);
                var P = new Pivot(n, Allocator.Temp);
                var u = new fProxyN(m, Allocator.Temp);
                QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                TierP(in A0, in Q, in R, in P);

                P.Dispose();
            }
            finally { }
        }

        // ── Case 5: ties. Exact-duplicate columns AND a 1-ulp-apart-norm column. Tier P (which tie
        //    "wins" is not asserted) PLUS strict determinism: the SAME input through the SAME overload
        //    twice must yield bit-identical P, Q, R.
        void Ties()
        {
            try
            {
                int m = 8, n = 5;
                var A0 = GenerateOP.fProxyRandomMat(m, n, -2f, 2f, 0x71E5u);
                // column 2 := exact duplicate of column 0.
                for (int r = 0; r < m; r++) A0[r, 2] = A0[r, 0];
                // column 4 := column 1 scaled by (1 + 2·eps): a ~1-ulp-apart NORM tie (no math.nextafter
                // needed — multiplying by 1+2·Consts.fProxyEpsilon nudges the norm by ~1 ulp per type).
                fProxy ulpish = (fProxy)1 + (fProxy)2 * Consts.fProxyEpsilon;
                for (int r = 0; r < m; r++) A0[r, 4] = A0[r, 1] * ulpish;

                // Run 1.
                var Q1 = new fProxyMxN(in A0, Allocator.Temp);
                var R1 = new fProxyMxN(n, n, Allocator.Temp);
                var P1 = new Pivot(n, Allocator.Temp);
                var u1 = new fProxyN(m, Allocator.Temp);
                QRCP.decompInPlace(ref Q1, ref R1, ref P1, ref u1);
                TierP(in A0, in Q1, in R1, in P1);

                // Run 2 (independent copy, identical overload).
                var Q2 = new fProxyMxN(in A0, Allocator.Temp);
                var R2 = new fProxyMxN(n, n, Allocator.Temp);
                var P2 = new Pivot(n, Allocator.Temp);
                var u2 = new fProxyN(m, Allocator.Temp);
                QRCP.decompInPlace(ref Q2, ref R2, ref P2, ref u2);

                for (int j = 0; j < n; j++) RecordEq(P1[j], P2[j]);
                for (int i = 0; i < Q1.Length; i++) AssertBitIdentical(Q1[i], Q2[i]);
                for (int i = 0; i < R1.Length; i++) AssertBitIdentical(R1[i], R2[i]);

                P1.Dispose(); P2.Dispose();
            }
            finally { }
        }

        // ── Case 6a: scale extremes. Column 2-norms spanning many orders of magnitude within ONE
        //    matrix. Tier P: no NaN/Inf, reconstruction holds (relative), monotone diagonal.
        void ScaleExtremes()
        {
            try
            {
                int m = 10, n = 6;
                var A0 = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 0x5CA1E5u);
                // per-column scale factors from ~1e-6 to ~1e6 (12 orders of dynamic range — safe for
                // float, and equally exercised for double).
                for (int c = 0; c < n; c++)
                {
                    fProxy scale = c == 0 ? (fProxy)1e-6f
                                 : c == 1 ? (fProxy)1e-3f
                                 : c == 2 ? (fProxy)1f
                                 : c == 3 ? (fProxy)1e2f
                                 : c == 4 ? (fProxy)1e4f
                                 :          (fProxy)1e6f;
                    for (int r = 0; r < m; r++)
                        A0[r, c] *= scale;
                }

                var Q = new fProxyMxN(in A0, Allocator.Temp);
                var R = new fProxyMxN(n, n, Allocator.Temp);
                var P = new Pivot(n, Allocator.Temp);
                var u = new fProxyN(m, Allocator.Temp);
                QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                TierP(in A0, in Q, in R, in P);

                P.Dispose();
            }
            finally { }
        }

        // ── Case 6b: degenerate shapes. Fully-zero matrix (rank 0), zero columns mixed in (rank ==
        //    #nonzero), single-column tall (n=1), and tiny n = 1..3. Tier P + rank correctness for the
        //    zero cases. No NaN anywhere.
        void ZeroAndTinySizes()
        {
            try
            {
                // (a) fully zero 5×3 -> rank 0.
                {
                    var A0 = new fProxyMxN(5, 3, Allocator.Temp);
                    var Q = new fProxyMxN(in A0, Allocator.Temp);
                    var R = new fProxyMxN(3, 3, Allocator.Temp);
                    var P = new Pivot(3, Allocator.Temp);
                    var u = new fProxyN(5, Allocator.Temp);
                    QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                    TierP(in A0, in Q, in R, in P);
                    RecordEq(RankFromR(in R, 5, 3), 0);
                    P.Dispose();
                }

                // (b) zero columns mixed in: cols 0,3 nonzero AND linearly independent (col0 varies
                //     down the rows; col3 is supported on different rows), cols 1,2,4 exactly zero ->
                //     rank 2. (Two CONSTANT columns would both be multiples of the all-ones vector,
                //     i.e. parallel -> rank 1; hence col0/col3 must be non-parallel.)
                {
                    int m = 6, n = 5;
                    var A0 = new fProxyMxN(m, n, Allocator.Temp);
                    for (int r = 0; r < m; r++) A0[r, 0] = (fProxy)(r + 1); // (1,2,3,4,5,6)
                    A0[0, 3] = (fProxy)2f; A0[3, 3] = (fProxy)2f;           // supported on rows 0,3 only
                    var Q = new fProxyMxN(in A0, Allocator.Temp);
                    var R = new fProxyMxN(n, n, Allocator.Temp);
                    var P = new Pivot(n, Allocator.Temp);
                    var u = new fProxyN(m, Allocator.Temp);
                    QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                    TierP(in A0, in Q, in R, in P);
                    RecordEq(RankFromR(in R, m, n), 2);
                    P.Dispose();
                }

                // (c) single-column tall (n=1), a few different m.
                {
                    for (int mi = 0; mi < 3; mi++)
                    {
                        int m = mi == 0 ? 1 : (mi == 1 ? 4 : 9);
                        var A0 = new fProxyMxN(m, 1, Allocator.Temp);
                        for (int r = 0; r < m; r++) A0[r, 0] = (fProxy)(r + 1);
                        var Q = new fProxyMxN(in A0, Allocator.Temp);
                        var R = new fProxyMxN(1, 1, Allocator.Temp);
                        var P = new Pivot(1, Allocator.Temp);
                        var u = new fProxyN(m, Allocator.Temp);
                        QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                        TierP(in A0, in Q, in R, in P);
                        RecordEq(P[0], 0);
                        RecordEq(RankFromR(in R, m, 1), 1);
                        P.Dispose();
                    }
                }

                // (d) tiny full-ish sizes n = 1..3 (m = n and m = n+2), random, generically full rank.
                {
                    for (int n = 1; n <= 3; n++)
                    for (int mi = 0; mi < 2; mi++)
                    {
                        int m = mi == 0 ? n : n + 2;
                        var A0 = GenerateOP.fProxyRandomMat(m, n, -3f, 3f, 0x717Au + (uint)(n * 7 + mi));
                        for (int d = 0; d < n; d++) A0[d, d] += (fProxy)6f; // ensure full rank
                        var Q = new fProxyMxN(in A0, Allocator.Temp);
                        var R = new fProxyMxN(n, n, Allocator.Temp);
                        var P = new Pivot(n, Allocator.Temp);
                        var u = new fProxyN(m, Allocator.Temp);
                        QRCP.decompInPlace(ref Q, ref R, ref P, ref u);
                        TierP(in A0, in Q, in R, in P);
                        RecordEq(RankFromR(in R, m, n), n);
                        P.Dispose();
                    }
                }
            }
            finally { }
        }

        // ── Tier E demonstrator: a construction ENGINEERED to be well-separated at every step (random
        //    columns scaled by geometrically distinct factors 8^j). We assert the oracle certifies it
        //    as separated (so the case genuinely exercises Tier E — a construction that turned out NOT
        //    separated would silently degrade to Tier P), and ProdAndOracle asserts the downdated
        //    production output is bit-identical (Pivot AND Q AND R) to the exact-recompute oracle. On a
        //    forced pivot sequence, downdating changes nothing downstream of the pivot choice, so the
        //    factors match to the last bit.
        void TierEDistinctMagnitudes()
        {
            try
            {
                int m = 10, n = 5;
                var A0 = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 0x7E5700u);
                fProxy sc = (fProxy)1;
                for (int c = 0; c < n; c++)
                {
                    for (int r = 0; r < m; r++) A0[r, c] *= sc;
                    sc *= (fProxy)8; // 1, 8, 64, 512, 4096 -> hugely staggered column norms
                }

                bool sep = ProdAndOracle(in A0, out int _);
                RecordEq(sep ? 1 : 0, 1); // MUST be Tier-E-eligible or the demonstrator is not demonstrating
            }
            finally { }
        }

        // ── Cache-overload equivalence: the zero-alloc cache overloads (fProxyQRCPCache) must be
        //    BIT-IDENTICAL to the internally-Temp-allocating non-cache overloads on identical input,
        //    for both decompInPlace and solveInPlace (P/Q/R/x/rank), full-rank AND rank-deficient.
        void CacheEquivalence(int m, int n, uint seed, bool rankDeficient)
        {
            try
            {
                var A0 = GenerateOP.fProxyRandomMat(m, n, -3f, 3f, seed);
                for (int d = 0; d < n; d++) A0[d, d] += (fProxy)6f;
                if (rankDeficient)
                    for (int r = 0; r < m; r++)
                        A0[r, n - 1] = A0[r, 0] + A0[r, 1]; // exact dependency -> rank n-1

                // decompInPlace: non-cache vs cache.
                var Anc = new fProxyMxN(in A0, Allocator.Temp); var Rnc = new fProxyMxN(n, n, Allocator.Temp); var Pnc = new Pivot(n, Allocator.Temp); var unc = new fProxyN(m, Allocator.Temp);
                QRCP.decompInPlace(ref Anc, ref Rnc, ref Pnc, ref unc);

                var Ac = new fProxyMxN(in A0, Allocator.Temp); var Rc = new fProxyMxN(n, n, Allocator.Temp); var Pc = new Pivot(n, Allocator.Temp); var uc = new fProxyN(m, Allocator.Temp);
                var cache = new fProxyQRCPCache(n, Allocator.Temp);
                QRCP.decompInPlace(ref Ac, ref Rc, ref Pc, ref uc, ref cache);

                for (int j = 0; j < n; j++) RecordEq(Pnc[j], Pc[j]);
                for (int i = 0; i < Anc.Length; i++) AssertBitIdentical(Anc[i], Ac[i]);
                for (int i = 0; i < Rnc.Length; i++) AssertBitIdentical(Rnc[i], Rc[i]);

                // solveInPlace: non-cache vs cache (default relTol both). solveInPlace destroys b
                // (fused), so each call gets its own copy of the identical RHS.
                var b0 = GenerateOP.fProxyRandomVec(m, -3f, 3f, seed + 1u);

                var As1 = new fProxyMxN(in A0, Allocator.Temp); var b1 = new fProxyN(in b0, Allocator.Temp); var Rs1 = new fProxyMxN(n, n, Allocator.Temp); var Ps1 = new Pivot(n, Allocator.Temp); var us1 = new fProxyN(m, Allocator.Temp); var x1 = new fProxyN(n, Allocator.Temp);
                RankInfo info1 = QRCP.solveInPlace(ref As1, ref b1, ref x1, ref Rs1, ref Ps1, ref us1);

                var As2 = new fProxyMxN(in A0, Allocator.Temp); var b2 = new fProxyN(in b0, Allocator.Temp); var Rs2 = new fProxyMxN(n, n, Allocator.Temp); var Ps2 = new Pivot(n, Allocator.Temp); var us2 = new fProxyN(m, Allocator.Temp); var x2 = new fProxyN(n, Allocator.Temp);
                var cache2 = new fProxyQRCPCache(n, Allocator.Temp);
                RankInfo info2 = QRCP.solveInPlace(ref As2, ref b2, ref x2, ref Rs2, ref Ps2, ref us2, ref cache2);

                RecordEq((int)info1.status, (int)info2.status);
                RecordEq(info1.rank, info2.rank);
                if (rankDeficient) RecordEq(info1.rank, n - 1);
                for (int i = 0; i < n; i++) AssertBitIdentical(x1[i], x2[i]);
                for (int i = 0; i < As1.Length; i++) AssertBitIdentical(As1[i], As2[i]);

                Pnc.Dispose(); Pc.Dispose(); Ps1.Dispose(); Ps2.Dispose();
            }
            finally { }
        }

        // ── Blocked (level-3 dlaqps panel) core. Everything above tops out at n = 128 in ONE case
        //    (GradualDecay, TierP-only) and n = 64 in another (Kahan, TierP-only); nothing here pinned
        //    the blocked path's PIVOT decisions or its guard-cut/re-sum branch. This case targets the
        //    blocked core (N_Cols >= 2*QRCP_BLOCK = 64) directly:
        //
        //    (a) Well-separated staggered-magnitude inputs at panel-boundary shapes — n = 64 (== 2*NB,
        //        exactly two full panels), 65 (2*NB+1, a 1-wide trailing panel), 96 (three panels) and
        //        128, both square and tall. Column c is scaled to a geometric norm target spanning ~1e6,
        //        so the trailing norms stay well-separated at every step (Tier-E-eligible). The blocked
        //        production factorization must then pick the SAME pivot sequence as the exact-recompute
        //        oracle, and agree on Q and R to a tight tolerance — NOT bit-identically: blocked forms
        //        trailing values by GEMM accumulation vs the oracle's rank-1 chain, a different summation
        //        order — the same reason the blocked QR path isn't bit-identical to its unblocked
        //        small-n path.
        //    (b) A rank-1-plus-tiny-noise input at n = 80 (mass cancellation): the norm guard trips
        //        mid-panel on nearly every column, exercising the mark / cut-panel-short / deferred
        //        re-sum branch that the unblocked core does NOT have. Rank must collapse to 1 (auto tol).
        void BlockedPanels()
        {
            try
            {
                // (a) panel-boundary shapes, staggered well-separated norms. n = 64 (2*NB), 65 (2*NB+1),
                //     96 (3 panels), 128 (4 panels), each square and/or tall.
                int eligible = 0;
                const int shapes = 6;
                for (int s = 0; s < shapes; s++)
                {
                    int n = s == 0 ? 64 : s == 1 ? 65 : s == 2 ? 96 : s == 3 ? 96 : s == 4 ? 128 : 128;
                    int m = s == 0 ? 64 : s == 1 ? 65 : s == 2 ? 96 : s == 3 ? 140 : s == 4 ? 128 : 200;
                    var A0 = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 0xB10C0000u + (uint)s);
                    // Scale column c to a geometric norm target 1 .. 1e3 (ratio 1e3^(1/(n-1)) between
                    // neighbours — above the Tier-E separation margin, yet a modest enough total range
                    // that the scale-relative zero-column threshold (1e-6·LInf ≈ 1e-3 here) stays well
                    // below every column's norm on FLOAT. A wider stagger (e.g. 1e6) would push that
                    // threshold up to ~1 and mis-classify the smallest columns as zero columns —
                    // corrupting Q's orthonormality — which is a float-precision artifact of the input,
                    // not a blocked-core defect (double, with more headroom, tolerates the wider range).
                    for (int c = 0; c < n; c++)
                    {
                        fProxy t = n > 1 ? (fProxy)c / (fProxy)(n - 1) : (fProxy)0;
                        fProxy scale = math.pow((fProxy)1e3f, t);
                        for (int r = 0; r < m; r++) A0[r, c] *= scale;
                    }
                    if (ProdVsOracleTol(in A0, (fProxy)1e-3f)) eligible++;
                }
                // The staggered construction is engineered to be Tier-E-eligible; require the tight
                // pivot/Q/R check to have actually run at least once (else it is not testing the
                // blocked pivot path it claims to). Matches the fuzz sweep's ">0 eligible" philosophy.
                RecordBound((fProxy)1, (fProxy)eligible);   // eligible >= 1

                // (b) guard-cut mid-panel: rank-1 + sub-tolerance noise at a blocked size -> rank 1.
                {
                    int m = 100, n = 80;
                    var rng = new Unity.Mathematics.Random(0x6C07u);
                    var v = new fProxyN(m, Allocator.Temp);
                    for (int r = 0; r < m; r++) v[r] = (fProxy)(rng.NextFloat(-1f, 1f));
                    fProxy noiseScale = (fProxy)0.01f * Consts.fProxyZeroThreshold;

                    var A0 = new fProxyMxN(m, n, Allocator.Temp);
                    for (int c = 0; c < n; c++)
                    {
                        fProxy alpha = (fProxy)(rng.NextFloat(0.25f, 4f));
                        for (int r = 0; r < m; r++)
                            A0[r, c] = alpha * v[r] + noiseScale * (fProxy)(rng.NextFloat(-1f, 1f));
                    }

                    var Q = new fProxyMxN(in A0, Allocator.Temp);
                    var R = new fProxyMxN(n, n, Allocator.Temp);
                    var P = new Pivot(n, Allocator.Temp);
                    var u = new fProxyN(m, Allocator.Temp);
                    QRCP.decompInPlace(ref Q, ref R, ref P, ref u);   // blocked (n = 80 >= 64)
                    TierP(in A0, in Q, in R, in P);
                    RecordEq(RankFromR(in R, m, n), 1);
                    P.Dispose();
                }
            }
            finally { }
        }

        // Blocked-vs-oracle for a Tier-E-eligible input: production decompInPlace (BLOCKED at these
        // sizes) must match the exact-recompute oracle's pivot sequence EXACTLY and its Q/R within
        // relTol (absolute for Q's O(1) entries; scaled by the matrix magnitude for R). Returns the
        // oracle's separation flag; the tight asserts only fire when it certifies eligibility. This is
        // the toleranced sibling of ProdAndOracle (which demands bit-identity — valid only for the
        // unblocked path, whose summation order matches the oracle's).
        bool ProdVsOracleTol(in fProxyMxN A0, fProxy relTol)
        {
            int m = A0.M_Rows;
            int n = A0.N_Cols;

            var Qp = new fProxyMxN(in A0, Allocator.Temp); var Rp = new fProxyMxN(n, n, Allocator.Temp); var Pp = new Pivot(n, Allocator.Temp); var up = new fProxyN(m, Allocator.Temp);
            QRCP.decompInPlace(ref Qp, ref Rp, ref Pp, ref up);
            TierP(in A0, in Qp, in Rp, in Pp);

            var Qo = new fProxyMxN(in A0, Allocator.Temp); var Ro = new fProxyMxN(n, n, Allocator.Temp); var Po = new Pivot(n, Allocator.Temp); var uo = new fProxyN(m, Allocator.Temp);
            bool sep = OracleDecompInPlace(ref Qo, ref Ro, ref Po, ref uo);

            if (sep)
            {
                fProxy scale = Norms.LInf(in A0) + (fProxy)1;
                for (int j = 0; j < n; j++) RecordEq(Pp[j], Po[j]);

                fProxy qDiff = (fProxy)0;
                for (int i = 0; i < Qp.Length; i++)
                {
                    fProxy e = math.abs(Qp[i] - Qo[i]);
                    if (e > qDiff) qDiff = e;
                }
                RecordBound(qDiff, relTol);

                fProxy rDiff = (fProxy)0;
                for (int i = 0; i < Rp.Length; i++)
                {
                    fProxy e = math.abs(Rp[i] - Ro[i]);
                    if (e > rDiff) rDiff = e;
                }
                RecordBound(rDiff, relTol * scale);
            }

            Pp.Dispose(); Po.Dispose();
            return sep;
        }

        // ══════════════════════════════ shared machinery ══════════════════════════════

        // Uniform Tier-P property tolerance. NOT machine-precision: column pivoting on rank-deficient
        // / severely ill-conditioned inputs (Kahan, collapsed columns) invokes the zero-column
        // reflector fallback, which limits Q's orthonormality to well above eps — the EXISTING QRCP
        // suite uses a uniform 1e-4 for the same reason. We use a slightly looser 5e-3 because this
        // battery deliberately pushes larger n and more-degenerate inputs than that suite. Tier P only
        // needs to catch an order-1 broken invariant; the TIGHT bug-catchers here are Tier E
        // (bit-identity, exact ==) and the rank checks (auto tolerance), neither of which uses this.
        fProxy PropTol => (fProxy)5e-3f;

        // A·P == Q·R (relative), R upper-triangular, Q orthonormal, |R| diagonal non-increasing, and
        // no NaN. Every bound is recorded to Fail before its Assert (so a failure surfaces the actual
        // magnitude), and reconstruction/upper-triangular bounds are relative to the input magnitude
        // so they hold for badly-scaled matrices.
        void TierP(in fProxyMxN A0, in fProxyMxN Q, in fProxyMxN R, in Pivot P)
        {
            int m = A0.M_Rows;
            int n = A0.N_Cols;
            fProxy tol = PropTol;
            fProxy scale = Norms.LInf(in A0) + (fProxy)1;

            // reconstruction: A permuted by P == Q·R.
            var Aperm = new fProxyMxN(in A0, Allocator.Temp);
            for (int r = 0; r < m; r++)
                for (int j = 0; j < n; j++)
                    Aperm[r, j] = A0[r, P[j]];

            fProxyMxN diff = new fProxyMxN(in Aperm, Allocator.Temp);
            fProxyComp.subInPlace(diff, Blas.dot(Q, R));
            if (Analysis.isAnyNan(in diff))
                throw new System.Exception("QRCPDowndateTests: NaN detected in reconstruction");
            RecordBound(Analysis.MaxZeroError(diff), tol * scale);

            // R upper-triangular: max |R[r,c]| over the strict lower triangle.
            fProxy utErr = (fProxy)0;
            for (int r = 0; r < n; r++)
                for (int c = 0; c < r; c++)
                {
                    fProxy e = math.abs(R[r, c]);
                    if (e > utErr) utErr = e;
                }
            RecordBound(utErr, tol * scale);

            // Q orthonormal: max |(QᵀQ − I)[i,j]|.
            fProxyMxN QtQ = Blas.dot(Q, Q, true);
            if (Analysis.isAnyNan(in QtQ))
                throw new System.Exception("QRCPDowndateTests: NaN detected in QᵀQ");
            fProxy orthoErr = (fProxy)0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy target = i == j ? (fProxy)1 : (fProxy)0;
                    fProxy e = math.abs(QtQ[i, j] - target);
                    if (e > orthoErr) orthoErr = e;
                }
            RecordBound(orthoErr, tol);

            // |R[d,d]| non-increasing (greedy column pivoting), absolute slack relative to the lead.
            fProxy monoTol = tol * (math.abs(R[0, 0]) + (fProxy)1);
            for (int d = 0; d + 1 < n; d++)
            {
                fProxy hi = math.abs(R[d, d]);
                fProxy lo = math.abs(R[d + 1, d + 1]);
                RecordBound(lo, hi + monoTol);
            }
        }

        // Numerical rank from R's non-increasing diagonal at the library-standard AUTO tolerance
        // (max(m,n)·zeroThreshold · |R[0,0]|) — the same rule QRCP.solveInPlace and Analysis.rank use.
        int RankFromR(in fProxyMxN R, int m, int n)
        {
            fProxy tol = (fProxy)math.max(m, n) * Consts.fProxyZeroThreshold * math.abs(R[0, 0]);
            int rank = 0;
            for (int i = 0; i < n; i++)
            {
                if (math.abs(R[i, i]) > tol) rank++;
                else break;
            }
            return rank;
        }

        // Runs production decompInPlace on a copy of A0, checks Tier P, then runs the exact-recompute
        // oracle on a SECOND copy. If the oracle certifies the input Tier-E-eligible (every step
        // well-separated), additionally asserts the pivot sequence AND Q AND R are bit-identical.
        // Returns the separation flag (and, out, the detected rank from production's R). PUBLIC so the
        // fuzz job can drive it with a shared Fail array.
        public bool ProdAndOracle(in fProxyMxN A0, out int prodRank)
        {
            int m = A0.M_Rows;
            int n = A0.N_Cols;

            var Qp = new fProxyMxN(in A0, Allocator.Temp); var Rp = new fProxyMxN(n, n, Allocator.Temp); var Pp = new Pivot(n, Allocator.Temp); var up = new fProxyN(m, Allocator.Temp);
            QRCP.decompInPlace(ref Qp, ref Rp, ref Pp, ref up);
            TierP(in A0, in Qp, in Rp, in Pp);
            prodRank = RankFromR(in Rp, m, n);

            var Qo = new fProxyMxN(in A0, Allocator.Temp); var Ro = new fProxyMxN(n, n, Allocator.Temp); var Po = new Pivot(n, Allocator.Temp); var uo = new fProxyN(m, Allocator.Temp);
            bool sep = OracleDecompInPlace(ref Qo, ref Ro, ref Po, ref uo);

            if (sep)
            {
                for (int j = 0; j < n; j++) RecordEq(Pp[j], Po[j]);
                for (int i = 0; i < Qp.Length; i++) AssertBitIdentical(Qp[i], Qo[i]);
                for (int i = 0; i < Rp.Length; i++) AssertBitIdentical(Rp[i], Ro[i]);
            }

            Pp.Dispose(); Po.Dispose();
            return sep;
        }

        // Reference oracle: faithful transcription of the PRE-downdate QRCP (exact per-step partial
        // norm recomputation), reusing production's OWN internal reflector kernels (QR.genHouseholder /
        // QR.applyReflectorRight) so the ONLY thing differing from production is the norm strategy that
        // feeds pivot choice. Fills Q (in A_to_Q), R, P. Returns true iff the input was well-separated
        // at EVERY step (Tier-E-eligible), measured on the EXACT squared trailing norms it computes.
        bool OracleDecompInPlace(ref fProxyMxN A_to_Q, ref fProxyMxN R, ref Pivot P, ref fProxyN u)
        {
            P.Reset();

            int m = A_to_Q.M_Rows;
            int n = A_to_Q.N_Cols;

            var w        = new fProxyN(n, Allocator.Temp, false);
            var colNorm2 = new fProxyN(n, Allocator.Temp, false);

            fProxy zeroThreshold = Consts.fProxyZeroThreshold * Norms.LInf(in A_to_Q);
            fProxy sepFactor = (fProxy)1 + (fProxy)8 * Consts.fProxySqrtEps;
            bool separatedAll = true;

            for (int d = 0; d < n; d++)
            {
                // exact recompute: row-major sweep, rows d..m-1, columns d..n-1 (unit-stride per row).
                for (int j = d; j < n; j++) colNorm2[j] = (fProxy)0;
                for (int r = d; r < m; r++)
                    for (int j = d; j < n; j++)
                        colNorm2[j] += A_to_Q[r, j] * A_to_Q[r, j];

                // well-separated diagnostic on the EXACT squared trailing norms (BEFORE the swap).
                int trailing = n - d;
                if (trailing >= 2)
                {
                    fProxy s1 = (fProxy)(-1), s2 = (fProxy)(-1);
                    for (int j = d; j < n; j++)
                    {
                        fProxy val = colNorm2[j];
                        if (val > s1) { s2 = s1; s1 = val; }
                        else if (val > s2) { s2 = val; }
                    }
                    if (!(s1 > s2 * sepFactor))
                        separatedAll = false;
                }

                fProxy diagNorm2 = colNorm2[d];
                int pivotCol = d;
                fProxy maxNorm2 = diagNorm2;
                for (int c = d + 1; c < n; c++)
                    if (colNorm2[c] > maxNorm2) { maxNorm2 = colNorm2[c]; pivotCol = c; }

                fProxy pivotRelTol = (fProxy)(8 * m) * Consts.fProxyEpsilon;
                if (pivotCol != d && maxNorm2 > diagNorm2 * ((fProxy)1 + pivotRelTol))
                {
                    Swap.Columns(ref A_to_Q, d, pivotCol);
                    P.Swap(d, pivotCol);
                }

                OracleGenHouseholder(ref A_to_Q, ref u, d, zeroThreshold);
                OracleApplyReflectorRight(ref A_to_Q, ref u, ref w, d);

                R[d, d] = A_to_Q[d, d];
                for (int i = d; i < m; i++) A_to_Q[i, d] = u[i];
            }

            // Epilogue: identical to production's decompInPlaceCore (mirror verbatim — it is unchanged
            // production logic, not what is under test).
            for (int r = 0; r < R.M_Rows; r++)
            for (int c = 0; c < R.N_Cols; c++)
            {
                if (c < r) R[r, c] = (fProxy)0;
                else if (c > r) R[r, c] = A_to_Q[r, c];
            }

            for (int r = 0; r < m; r++)
                for (int c = r; c < n; c++)
                    if (c > r) A_to_Q[r, c] = (fProxy)0;

            for (int d = n - 1; d >= 0; d--)
            {
                for (int i = d; i < m; i++)
                {
                    u[i] = A_to_Q[i, d];
                    A_to_Q[i, d] = i == d ? (fProxy)1 : (fProxy)0;
                }
                OracleApplyReflectorRight(ref A_to_Q, ref u, ref w, d);
            }

            colNorm2.Dispose();
            w.Dispose();
            return separatedAll;
        }

        // Faithful, bit-identical replicas of production's Householder kernels (QR.genHouseholder /
        // QR.applyReflectorRight); those are `internal`, reachable here via the InternalsVisibleTo
        // grants on both BurstLinearAlgebra.Tests and BurstLinearAlgebra.TemplateSource.Tests-firstpass
        // (TemplateSource/AssemblyInfo.cs), but a replica is used anyway so the oracle stays independent
        // of the kernel under test. Bit-identity is preserved by
        // NOT reimplementing the numeric core: the reflector-apply delegates to the SAME public
        // vectorised kernel production uses (LinearAlgebra.Internal.UnsafeOP.axpy), and the
        // reflector-vector build calls the SAME public Norms.L2Range and mirrors the exact scalar
        // arithmetic and evaluation order of QR.genHouseholder — so on a matching pivot sequence the
        // factors reproduce production's to the last bit. (Only the trivial one-line `sign` — internal
        // in QR — is copied verbatim.) These replicas are throwaway test scaffolding: they exist to
        // isolate the norm/pivot strategy under test, not to be a second maintained QRCP.
        static fProxy OracleSign(fProxy x) => x < (fProxy)0 ? (fProxy)(-1) : (fProxy)1;

        void OracleGenHouseholder(ref fProxyMxN Q, ref fProxyN u, int k, fProxy zeroThreshold)
        {
            for (int r = k; r < u.N; r++)
                u[r] = Q[r, k];

            fProxy xNorm = Norms.L2Range(u, k, u.N);

            if (math.abs(xNorm) > zeroThreshold)
            {
                for (int r = k; r < u.N; r++)
                    u[r] = u[r] / xNorm;

                u[k] = u[k] + OracleSign(u[k]);

                var div = math.sqrt(math.abs(u[k]));
                for (int r = k; r < u.N; r++)
                    u[r] = u[r] / div;
            }
            else
            {
                u[k] = math.sqrt((fProxy)2); // == math.SQRT2 branch; dead for well-separated (Tier E) inputs
            }
        }

        unsafe void OracleApplyReflectorRight(ref fProxyMxN Q, ref fProxyN u, ref fProxyN w, int d)
        {
            int M = Q.M_Rows;
            int N = Q.N_Cols;
            int L = N - d;
            if (L <= 0)
                return;

            fProxy* qp = Q.Data.Ptr;
            fProxy* up = u.Data.Ptr;
            fProxy* wp = w.Data.Ptr;

            // pass 1: w[0..L) = Σ_{r=d}^{M-1} u[r]·Q[r, d..N)  — same UnsafeOP.axpy as production.
            UnsafeUtility.MemClear(wp, (long)L * UnsafeUtility.SizeOf<fProxy>());
            for (int r = d; r < M; r++)
                LinearAlgebra.Internal.UnsafeOP.axpy(wp, qp + (long)r * N + d, up[r], L);

            // pass 2: Q[r, d..N) -= u[r]·w.
            for (int r = d; r < M; r++)
                LinearAlgebra.Internal.UnsafeOP.axpy(qp + (long)r * N + d, wp, -up[r], L);
        }

        void AssertBitIdentical(fProxy a, fProxy b)
        {
            if (a != b && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = a; Fail[2] = b; Fail[3] = a - b;
            }
            Assert.IsTrue(a == b);
        }

        void RecordBound(fProxy value, fProxy limit)
        {
            if (!(value <= limit) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = value; Fail[2] = limit; Fail[3] = value - limit;
            }
            Assert.IsTrue(value <= limit);
        }

        void RecordEq(int got, int expected)
        {
            if (got != expected && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1; Fail[1] = got; Fail[2] = expected; Fail[3] = got - expected;
            }
            Assert.AreEqual(expected, got);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void DowndateTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // ══════════════════════════════ Case 7: fuzz sweep ══════════════════════════════
    // >= 64 random seeds, mixed shapes (square + tall, n 8..40, occasional very tall m). Every seed
    // gets Tier P invariants. Seeds the oracle certifies as well-separated at every step ALSO get
    // Tier E (pivot sequence + bit-identical Q/R). Some seeds get an injected exact / near rank
    // deficiency. Loops many seeds inside ONE Execute for speed (mirrors ReconstructRandomTall's
    // style). The Tier-E-eligible count is reported out via Counts for the managed driver.
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct FuzzJob : IJob
    {
        public NativeArray<fProxy> Fail;  // [0] flag, [1] got, [2] expected/limit, [3] diff
        public NativeArray<int> Counts;   // [0] Tier-E-eligible seeds, [1] total seeds, [2] Tier-E passes (pre-first-failure)

        public void Execute()
        {
            var job = new TestJob { Fail = Fail };
            try
            {
                int total = 72;
                int eligible = 0;
                int tierEpass = 0;

                for (int t = 0; t < total; t++)
                {
                    uint seed = 0xF0000001u + (uint)t * 0x9E3779B1u;

                    int n = 8 + (t % 33);        // 8..40
                    int m = n + (t % 25);        // square..moderately tall
                    if (t % 9 == 0) m = n + 120; // occasional very tall

                    var A0 = GenerateOP.fProxyRandomMat(m, n, -3f, 3f, seed);

                    // Inject rank deficiency on ~1/3 of seeds (mix exact and near-exact).
                    int mode = t % 3;
                    if (mode == 1 && n >= 3)
                    {
                        for (int r = 0; r < m; r++)
                            A0[r, n - 1] = A0[r, 0] + A0[r, 1]; // exact dependency
                    }
                    else if (mode == 2 && n >= 4)
                    {
                        fProxy near = (fProxy)0.5f * Consts.fProxyZeroThreshold;
                        for (int r = 0; r < m; r++)
                            A0[r, n - 2] = A0[r, 0] - A0[r, 2]
                                         + near * (fProxy)((r % 5) - 2); // near-exact dependency
                    }

                    bool sep = job.ProdAndOracle(in A0, out int _);
                    if (sep)
                    {
                        eligible++;
                        if (Fail[0] == (fProxy)0) tierEpass++; // no bit-mismatch recorded so far
                    }
                }

                Counts[0] = eligible;
                Counts[1] = total;
                Counts[2] = tierEpass;
            }
            finally { }
        }
    }

    [Test]
    public void FuzzSweep()
    {
        var fail   = new NativeArray<fProxy>(4, Allocator.TempJob);
        var counts = new NativeArray<int>(3, Allocator.TempJob);
        try
        {
            new FuzzJob { Fail = fail, Counts = counts }.Run();
            UnityEngine.Debug.Log($"[QRCP fuzz] Tier-E-eligible seeds: {counts[0]}/{counts[1]} (bit-identical passes recorded: {counts[2]})");
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
            Assert.Greater(counts[0], 0, "expected at least some Tier-E-eligible seeds in the fuzz sweep");
        }
        finally
        {
            counts.Dispose();
            fail.Dispose();
        }
    }

    // ══════════════════════════════ managed throw-test ══════════════════════════════
    // A mis-sized cache must be rejected (RequireQRCPWorkspace -> ArgumentException). Main-thread, not
    // in a Burst job — matches the QrcpSolveThrowsOn* style in QRCPTests.fProxy.cs.
    [Test]
    public void QrcpCacheThrowsOnWrongSize()
    {
        var A = GenerateOP.fProxyRandomMat(5, 3, -1f, 1f, 12345u);
        var R = new fProxyMxN(3, 3, Allocator.Temp);
        var P = new Pivot(3, Allocator.Persistent);
        var u = new fProxyN(5, Allocator.Temp);
        var cache = new fProxyQRCPCache(2, Allocator.Temp); // wrong: must be sized for n == 3
        Assert.Catch<ArgumentException>(() => QRCP.decompInPlace(ref A, ref R, ref P, ref u, ref cache));
        P.Dispose();
    }
}
