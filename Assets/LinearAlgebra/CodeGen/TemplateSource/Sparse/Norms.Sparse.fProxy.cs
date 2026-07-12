using System;
using LinearAlgebra.Internal;
using LinearAlgebra.Sparse;

namespace LinearAlgebra
{
    public static partial class Norms
    {
        /// <summary>
        /// Entrywise L1 norm of a BSR matrix: Σ|aᵢⱼ| over the STORED entries. Implicit (absent)
        /// blocks contribute 0, so this equals the dense entrywise L1 of the expanded matrix.
        ///
        /// NOT supported for Symmetric (lower-block-triangle-only) storage: the implicit upper
        /// blocks are not materialized, so a single pass would under-count off-diagonals -- throws
        /// in that case.
        /// </summary>
        public static fProxy L1(in fProxyBSR A)
        {
            if (A.Symmetric)
                throw new ArgumentException("L1: not supported for Symmetric (lower-block-only) storage -- the implicit upper blocks would be under-counted");
            unsafe { return UnsafeOP.sumAbs(A.Values.Ptr, A.Values.Length); }
        }

        /// <summary>
        /// Entrywise L-infinity (max-abs) norm of a BSR matrix over the STORED entries; 0 for an
        /// empty matrix. Equals the dense entrywise max-abs of the expanded matrix (an implicit
        /// zero can never exceed a stored |value|).
        /// </summary>
        public static fProxy LInf(in fProxyBSR A)
        {
            var vals = A.Values;
            if (vals.Length == 0) return (fProxy)0;
            unsafe { return UnsafeOP.maxAbs(vals.Ptr, vals.Length); }
        }

        /// <summary>
        /// Frobenius (entrywise L2) norm of a BSR matrix: sqrt(Σ aᵢⱼ²) over the STORED entries —
        /// exact, since implicit zeros contribute nothing.
        ///
        /// NOT supported for Symmetric (lower-block-triangle-only) storage: the implicit upper
        /// blocks are not materialized, so a single pass would under-count off-diagonals -- throws
        /// in that case.
        /// </summary>
        public static fProxy L2(in fProxyBSR A)
        {
            if (A.Symmetric)
                throw new ArgumentException("L2: not supported for Symmetric (lower-block-only) storage -- the implicit upper blocks would be under-counted");
            unsafe
            {
                var vals = A.Values;
                return Unity.Mathematics.math.sqrt(UnsafeOP.vecDot(vals.Ptr, vals.Ptr, vals.Length));
            }
        }
    }
}
