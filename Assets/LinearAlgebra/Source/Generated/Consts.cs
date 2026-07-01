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
    }

}