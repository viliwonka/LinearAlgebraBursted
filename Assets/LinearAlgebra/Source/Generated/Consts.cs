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

        // Default PER-VALUE sweep/iteration budget for the SVD/Eigen QR-type diagonalizations
        // (bidiagonal QR, tridiagonal QL, Hessenberg QR) -- LAPACK dbdsqr's scaling (MAXITR=6,
        // i.e. an effective 6*n total across n values) rather than a flat constant: a flat cap
        // does not grow with the problem, so a large clustered/graded spectrum can legitimately
        // exhaust it (see docs/spec-svd-eigen-convergence.md). Floored at 75 (the library's
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