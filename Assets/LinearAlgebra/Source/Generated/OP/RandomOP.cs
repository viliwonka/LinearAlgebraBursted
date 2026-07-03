using System;
using System.Runtime.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebra
{
    /// <summary>
    /// Type-agnostic random permutation and shuffle operations over <see cref="Pivot"/> and
    /// <see cref="Indices"/> buffers. Not generated per-type — these operate on integer-index
    /// structures that are shared across all element types.
    /// </summary>
    public static class Random_OP
    {
        /// <summary>
        /// Resets <paramref name="p"/> to the identity permutation then shuffles it uniformly
        /// in place using Fisher–Yates (Knuth): for i = N−1 downto 1, swap p[i] with p[j]
        /// where j = NextInt(0, i+1). <see cref="Pivot.Swap"/> keeps the parity/Sign field
        /// correct automatically.
        /// A separate loop from <see cref="shuffleInpl"/> is intentional: Pivot.Swap tracks
        /// the permutation parity via its swap counter, which plain index swapping cannot do.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void randomPermutationInpl(ref Pivot p, ref Random rng)
        {
            p.Reset();
            for (int i = p.N - 1; i >= 1; i--)
            {
                int j = rng.NextInt(0, i + 1);
                p.Swap(i, j);
            }
        }

        /// <summary>
        /// Shuffles the existing contents of <paramref name="idx"/> in place using the same
        /// Fisher–Yates sweep as <see cref="randomPermutationInpl"/>, but does not reset or
        /// repopulate the buffer — the caller provides the initial contents.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void shuffleInpl(ref Indices idx, ref Random rng)
        {
            for (int i = idx.N - 1; i >= 1; i--)
            {
                int j = rng.NextInt(0, i + 1);
                int tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
        }

        /// <summary>
        /// Fills <paramref name="dest"/> with <c>dest.N</c> distinct indices chosen uniformly
        /// at random from <c>[0, n)</c> without replacement (Knuth partial Fisher–Yates).
        /// Allocates a temporary <c>Temp</c> scratch array of length n, initialises it to
        /// 0..n−1, performs dest.N partial swaps from the front to select the sample, copies
        /// results into <paramref name="dest"/>, and disposes the scratch before returning.
        /// If dest.N == 0 returns immediately without allocating the scratch.
        /// Throws if <paramref name="n"/> &lt;= 0 or <c>dest.N &gt; n</c>.
        /// Not decorated with AggressiveInlining because it allocates a scratch buffer.
        /// </summary>
        public static void sampleKWithoutReplacementInpl(ref Indices dest, int n, ref Random rng)
        {
            if (n <= 0)
                throw new ArgumentException("Random_OP.sampleKWithoutReplacementInpl: n must be > 0");
            int k = dest.N;
            if (k > n)
                throw new ArgumentException("Random_OP.sampleKWithoutReplacementInpl: dest.N must be <= n");

            if (k == 0) return; // nothing to sample; skip the n-length scratch allocation

            var scratch = new UnsafeList<int>(n, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            scratch.Resize(n, NativeArrayOptions.UninitializedMemory);

            for (int i = 0; i < n; i++)
                scratch[i] = i;

            for (int i = 0; i < k; i++)
            {
                int j = rng.NextInt(i, n);
                int tmp = scratch[i];
                scratch[i] = scratch[j];
                scratch[j] = tmp;
                dest[i] = scratch[i];
            }

            scratch.Dispose();
        }
    }
}
