//singularFile//

using System;

namespace LinearAlgebra
{
    // Type-agnostic Krylov helpers. This lives in a //singularFile// partial (emitted ONCE,
    // NOT multiplied into float/double) because RequireDistinctBuffers has no fProxy in its
    // signature: if it were declared in the multiplying Krylov.fProxy.cs template it would be
    // copied identically into both Krylov.float.cs and Krylov.double.cs -- two definitions of
    // the same member in the same partial class -> CS0111. See docs/dev/codegen-refactor-lessons.md.
    public static partial class Krylov
    {
        /// <summary>
        /// Throws <see cref="ArgumentException"/>(who) if any two of the first <paramref
        /// name="count"/> pointers in <paramref name="ptrs"/> are equal. Same "every scratch
        /// vector argument must be a distinct buffer" contract as cg/pcg's hand-expanded pairwise
        /// chains (see <c>cg&lt;TOp&gt;</c>'s guard comment for why: the elementwise scratch
        /// updates -- addScaledInPlace/scaleAddInPlace/etc. -- don't self-check aliasing, so silent
        /// corruption replaces a thrown exception). Expressed as a loop over a stack-allocated
        /// pointer array (rather than cg's/pcg's hand-written OR chain) because MINRES/BiCGSTAB/
        /// CGLS/LSQR carry 6-9 scratch vectors -- a hand-expanded chain would run 15-36 terms and
        /// become error-prone to write and review by hand. Pointers are compared as <c>long</c>
        /// (not <c>fProxy*</c> directly) purely to keep the stackalloc'd array a single-indirection
        /// primitive-typed buffer.
        /// </summary>
        static unsafe void RequireDistinctBuffers(string who, long* ptrs, int count)
        {
            for (int i = 0; i < count; i++)
                for (int j = i + 1; j < count; j++)
                    if (ptrs[i] == ptrs[j])
                        throw new ArgumentException(who);
        }
    }
}
