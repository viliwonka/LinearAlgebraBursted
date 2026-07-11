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
        public const int fProxyCholPivotBlockMinN = 256;
        public const int fProxyLuBlockMinN = 256;
        //-deleteThis
        public const float floatZeroThreshold = 1e-6f;
        public const float floatEpsilon = 1.1920929e-7f;   // machine epsilon, 2^-23
        public const float floatSqrtEps = 3.4526698e-4f;   // sqrt(floatEpsilon): best localization of a smooth minimum

        public const double doubleZeroThreshold = 1e-14; // could lower this, if necessary
        public const double doubleEpsilon = 2.220446049250313e-16;  // machine epsilon, 2^-52
        public const double doubleSqrtEps = 1.4901161193847656e-8;  // sqrt(doubleEpsilon): best localization of a smooth minimum

        // Row-count gate for LQ's blocked (compact-WY) vs unblocked kernel. Cache-dependent; measured
        // per dtype, pinned conservatively (err high) since a too-low gate can regress on a weaker cache.
        public const int floatLqBlockMinM  = 256;
        public const int doubleLqBlockMinM = 512;

        // Per-type level-3 blocking gates for the other factorizations, measured per dtype from a
        // blocked-vs-unblocked sweep (same convention as the LQ gate above). QR/QRCP gate on N_Cols
        // (column panels); Cholesky/LU gate on the matrix side n/m. float/double ordering is not
        // universal, so each is measured independently rather than derived. The fProxy* placeholders
        // above carry the template-compile default.
        public const int floatQrBlockMinN    = 128;
        public const int doubleQrBlockMinN   = 512;
        public const int floatQrcpBlockMinN  = 64;
        public const int doubleQrcpBlockMinN = 512;
        public const int floatCholBlockMinN  = 1024;
        public const int doubleCholBlockMinN = 512;
        public const int floatLuBlockMinN    = 256;
        public const int doubleLuBlockMinN   = 128;

        // Pivoted Cholesky (CHOP, xPSTRF) blocked-path gate; measured, same convention as the gates above.
        public const int floatCholPivotBlockMinN  = 512;
        public const int doubleCholPivotBlockMinN = 512;

        // Default PER-VALUE sweep/iteration budget for the SVD/Eigen QR-type diagonalizations
        // (bidiagonal QR, tridiagonal QL, Hessenberg QR) -- LAPACK dbdsqr's scaling (MAXITR=6,
        // i.e. an effective 6*n total across n values) rather than a flat constant: a flat cap
        // does not grow with the problem, so a large clustered/graded spectrum can legitimately
        // exhaust it. Floored at 75 (the library's original flat constant) so tiny problems keep
        // the same sane minimum they always had. `n` is whatever per-value dimension is actually
        // being iterated at each call site (the full matrix side for thin/values/valuesSymmetricInPlace/
        // symmetric/valuesQR; the smaller reduced-problem size for the GKL truncated/randomized SVD
        // routes) -- see each call site. This is a pathological-input BACKSTOP, not a target:
        // legitimate inputs should converge in a small fraction of it. Explicit maxIter/maxSweeps
        // arguments are never affected by this -- only the convenience overloads' hardcoded
        // defaults route through it.
        public static int sweepBudget(int n)
        {
            int scaled = 6 * n;
            return scaled > 75 ? scaled : 75;
        }
    }

}