//singularFile//

using System;

namespace LinearAlgebra
{
    // Type-agnostic Krylov helpers (singular partial -- not float/double multiplied).
    public static partial class Krylov
    {
        /// <summary>
        /// Throws if any two of the first <paramref name="count"/> pointers in <paramref
        /// name="ptrs"/> are equal (aliasing guard for solver scratch buffers).
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
