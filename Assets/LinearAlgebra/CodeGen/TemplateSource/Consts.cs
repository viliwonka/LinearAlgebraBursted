using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//singularFile//
namespace LinearAlgebra {

    public static class Consts {

        // Needed as literal members (not just deleteThis-stripped scratch): TemplateSource compiles
        // the raw .fProxy.cs files as-is, pre-substitution, as its own assembly - callers there write
        // Consts.fProxyZeroThreshold literally, so it must exist for THAT compile, even though codegen
        // strips this block from the generated (float/double) output in favor of the members below.
        //+deleteThis
        public const float fProxyZeroThreshold = 1e-6f;
        public const float fProxyEpsilon = 1.1920929e-7f;
        public const float fProxySqrtEps = 3.4526698e-4f;
        public const int fProxyLqBlockMinM = 512;
        public const int fProxyQrBlockMinN = 64;
        public const int fProxyQrcpBlockMinN = 64;
        public const int fProxyCholBlockMinN = 256;
        public const int fProxyLuBlockMinN = 256;
        //-deleteThis
        public const float floatZeroThreshold = 1e-6f;
        public const float floatEpsilon = 1.1920929e-7f;   // machine epsilon, 2^-23
        public const float floatSqrtEps = 3.4526698e-4f;   // sqrt(floatEpsilon): best localization of a smooth minimum

        public const double doubleZeroThreshold = 1e-14; // could lower this, if necessary
        public const double doubleEpsilon = 2.220446049250313e-16;  // machine epsilon, 2^-52
        public const double doubleSqrtEps = 1.4901161193847656e-8;  // sqrt(doubleEpsilon): best localization of a smooth minimum

        // Row-count gate above which LQ factorization switches from the unblocked (level-2) kernel to
        // the blocked compact-WY (level-3) core. MEASURED, cache-dependent crossover, not derived: it
        // is intrinsically CPU-specific (L2/L3 size, bandwidth-to-compute ratio), so these are pinned
        // CONSERVATIVELY (err HIGH). Below the gate the always-correct unblocked path runs, so a
        // too-high gate only forgoes upside, while a too-low gate can REGRESS on a weaker cache; a
        // worse CPU (smaller cache) crosses over EARLIER, so a high gate still captures its blocking
        // win. float and double differ because LQ's trailing-update fold is memory-reduction-bound and
        // double streams 2x the bytes per element, so double stays bandwidth-starved (blocking pays
        // off) only at a larger size: measured on this dev box float wins from ~256 row-panels, double
        // not until ~512. Tuned on TallWideSolveBenchmark (A is k x 2k).
        public const int floatLqBlockMinM  = 256;
        public const int doubleLqBlockMinM = 512;

        // Per-type level-3 blocking gates for the other factorizations (same rationale as LQ above:
        // MEASURED, cache-dependent, err HIGH). QR/QRCP gate on N_Cols (column panels), Cholesky/LU on
        // the matrix side n/m. Pinned from a same-session blocked-vs-unblocked sweep on the QR /
        // Cholesky / LU / QRVariants benchmarks (each value = the smallest swept size where the blocked
        // core actually beat the plain sweep for that type). The fProxy* placeholders above carry the
        // template-compile default.
        //
        // The float-vs-double ordering is NOT universal — it depends on where the bandwidth pressure
        // sits, which is why every gate was measured rather than derived:
        //   * QR / QRCP / Cholesky reconstruct-or-fold work is memory-reduction-bound, so DOUBLE (2x
        //     the bytes) stays starved and crosses over LATER -> higher double gate.
        //   * LU's trailing update is a proper GEMM; there the UNBLOCKED path is what re-streams the
        //     trailing matrix, and double's 2x traffic makes that hurt SOONER -> double crosses EARLIER
        //     (lower double gate than float). Opposite ordering, caught only because we benched.
        // Old shared gates (QR/QRCP 64, Cholesky/LU 256) were actively regressing double below its true
        // crossover (double QR at N=64 was ~40% slower blocked; Cholesky double at 256 ~15% slower).
        public const int floatQrBlockMinN    = 128;    // float wins from 128 (64 ~neutral)
        public const int doubleQrBlockMinN   = 512;    // double loses <=256, wins from 512
        public const int floatQrcpBlockMinN  = 64;     // float wins at every size
        public const int doubleQrcpBlockMinN = 512;    // double loses <=256, wins from 512
        public const int floatCholBlockMinN  = 1024;   // float loses <=512, wins from 1024
        public const int doubleCholBlockMinN = 512;    // double loses at 256, wins from 512
        public const int floatLuBlockMinN    = 256;    // float loses at 128, wins from 256
        public const int doubleLuBlockMinN   = 128;    // double wins from 128 (GEMM update: crosses earlier)

        // Default PER-VALUE sweep/iteration budget for the SVD/Eigen QR-type diagonalizations
        // (bidiagonal QR, tridiagonal QL, Hessenberg QR) -- LAPACK dbdsqr's scaling (MAXITR=6,
        // i.e. an effective 6*n total across n values) rather than a flat constant: a flat cap
        // does not grow with the problem, so a large clustered/graded spectrum can legitimately
        // exhaust it (see docs/dev/spec-svd-eigen-convergence.md). Floored at 75 (the library's
        // original flat constant) so tiny problems keep the same sane minimum they always had.
        // `n` is whatever per-value dimension is actually being iterated at each call site (the
        // full matrix side for thin/values/valuesSymmetric/symmetric/valuesQR; the smaller
        // reduced-problem size for the GKL truncated/randomized SVD routes) -- see each call site.
        // This is a pathological-input BACKSTOP, not a target: legitimate inputs should converge
        // in a small fraction of it. Explicit maxIter/maxSweeps arguments are never affected by
        // this -- only the convenience overloads' hardcoded defaults route through it.
        public static int sweepBudget(int n)
        {
            int scaled = 6 * n;
            return scaled > 75 ? scaled : 75;
        }
    }

}