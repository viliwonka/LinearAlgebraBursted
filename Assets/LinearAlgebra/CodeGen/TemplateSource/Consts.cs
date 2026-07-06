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
        //-deleteThis
        public const float floatZeroThreshold = 1e-6f;
        public const float floatEpsilon = 1.1920929e-7f;   // machine epsilon, 2^-23
        public const float floatSqrtEps = 3.4526698e-4f;   // sqrt(floatEpsilon): best localization of a smooth minimum

        public const double doubleZeroThreshold = 1e-14; // could lower this, if necessary
        public const double doubleEpsilon = 2.220446049250313e-16;  // machine epsilon, 2^-52
        public const double doubleSqrtEps = 1.4901161193847656e-8;  // sqrt(doubleEpsilon): best localization of a smooth minimum

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