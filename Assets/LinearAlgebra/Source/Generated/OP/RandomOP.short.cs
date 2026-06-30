using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebra
{
    /// <summary>
    /// Zero-alloc random-fill operations for integer vectors and matrices.
    /// Uniform refill: <c>nextUniformInpl</c> — overwrites a buffer directly from the caller's
    /// evolving RNG stream. Range is [min, max) per Unity's NextInt contract.
    ///
    /// <b>int-range limitation:</b> random draws use <c>Unity.Mathematics.Random.NextInt</c>,
    /// which takes <c>int</c> bounds. For the <c>long</c> expansion, min and max must lie
    /// within [int.MinValue, int.MaxValue]; values outside that range throw
    /// <see cref="ArgumentException"/>. This guard is always-false (harmless) for the
    /// <c>int</c> and <c>short</c> expansions where the type already fits.
    ///
    /// <b>min == max behaviour:</b> fills the buffer with that constant value.
    /// This is decoupled from the [min, max) range claim — it is simply a constant fill
    /// that avoids calling NextInt on the empty range [x, x), which is undefined
    /// in Unity.Mathematics.
    /// </summary>
    public static partial class shortRandom_OP
    {
        // ---- uniform refill (vector) ----

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from
        /// [<paramref name="min"/>, <paramref name="max"/>), advancing <paramref name="rng"/>
        /// by <c>dest.N</c> steps. Range is [min, max) per Unity NextInt.
        /// <para>If min == max the buffer is filled with that constant; no RNG advance occurs.</para>
        /// <para>Throws <see cref="ArgumentException"/> if min &gt; max, or (long expansion only)
        /// if min or max falls outside [int.MinValue, int.MaxValue].</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref shortN dest, short min, short max)
        {
            if (min < int.MinValue || max > int.MaxValue)
                throw new ArgumentException("shortRandom_OP: min/max must be within int range for random generation");
            if (!(min <= max))
                throw new ArgumentException("shortRandom_OP: min must be <= max");
            int len = dest.Data.Length;
            if (min == max)
            {
                for (int i = 0; i < len; i++)
                    dest[i] = min;
            }
            else
            {
                for (int i = 0; i < len; i++)
                    dest[i] = (short)rng.NextInt((int)min, (int)max);
            }
        }

        // ---- uniform refill (matrix) ----

        /// <summary>
        /// Overwrites every element of <paramref name="dest"/> with a uniform draw from
        /// [<paramref name="min"/>, <paramref name="max"/>), advancing <paramref name="rng"/>
        /// by <c>dest.Length</c> steps. Range is [min, max) per Unity NextInt.
        /// <para>If min == max the buffer is filled with that constant; no RNG advance occurs.</para>
        /// <para>Throws <see cref="ArgumentException"/> if min &gt; max, or (long expansion only)
        /// if min or max falls outside [int.MinValue, int.MaxValue].</para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void nextUniformInpl(ref Random rng, ref shortMxN dest, short min, short max)
        {
            if (min < int.MinValue || max > int.MaxValue)
                throw new ArgumentException("shortRandom_OP: min/max must be within int range for random generation");
            if (!(min <= max))
                throw new ArgumentException("shortRandom_OP: min must be <= max");
            int len = dest.Data.Length;
            if (min == max)
            {
                for (int i = 0; i < len; i++)
                    dest[i] = min;
            }
            else
            {
                for (int i = 0; i < len; i++)
                    dest[i] = (short)rng.NextInt((int)min, (int)max);
            }
        }
    }
}
