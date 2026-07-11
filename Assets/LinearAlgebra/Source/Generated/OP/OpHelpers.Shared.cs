using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Type-agnostic helpers shared by the FFT / Resample templates. A merged
    // partial class (float+double emit the same `FFT`/`Resample`) cannot hold the same int-only
    // signature twice, so these live in this single non-templated file and emit exactly once.
    public static partial class FFT
    {
        static bool IsPow2(int n) => n > 0 && (n & (n - 1)) == 0;

        static bool IsPowerOf4(int n) =>
            (n > 0) && ((n & (n - 1)) == 0) && ((n & unchecked((int)0xAAAAAAAA)) == 0);

        // Base-4 digit reversal: reverse the log4n base-4 digits of x.
        // Equivalent to reversing bit-pairs (the 2 bits within each pair stay in order).
        static int ReverseBase4Digits(int x, int log4n)
        {
            int result = 0;
            for (int d = 0; d < log4n; d++)
            {
                result = (result << 2) | (x & 3);
                x >>= 2;
            }
            return result;
        }
    }

    public static partial class Resample
    {
        // Resolves an arbitrary integer index i into [0, n-1] per EdgeMode. Clamp is the cheap
        // common path (math.clamp); Wrap/Mirror use modulo.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int idx(int i, int n, EdgeMode edge)
        {
            switch (edge)
            {
                case EdgeMode.Clamp:
                    return math.clamp(i, 0, n - 1);
                case EdgeMode.Wrap:
                    return ((i % n) + n) % n;
                default: // EdgeMode.Mirror — no-edge-repeat, period 2*(n-1)
                    if (n == 1) return 0;
                    int p = 2 * (n - 1);
                    int iMod = ((i % p) + p) % p;
                    return iMod < n ? iMod : p - iMod;
            }
        }
    }
}
