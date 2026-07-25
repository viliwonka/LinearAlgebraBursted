using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace BULA.Internal
{
    // Scalar helpers shared by the Householder/QL/bidiagonal kernels (QR, LQ, Bidiag, SVD, Eigen).
    internal static partial class Helpers
    {
        // |a| with the sign of b (Fortran SIGN / C99 copysign, NR convention): b >= 0 -> +|a|,
        // INCLUDING b == 0. Deliberately not math.sign-based: math.sign(0) == 0, which would zero
        // out the Householder sign choice (copysign(1, 0) must be +1 so a reflector built on a
        // zero pivot entry still gets a nonzero shift).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy copysign(fProxy a, fProxy b) => b >= (fProxy)0 ? math.abs(a) : -math.abs(a);

        // +1 or -1 with the sign of x, zero -> +1 (the Householder sign choice). Same zero
        // convention as copysign above; a direct branch, no abs.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy signOrOne(fProxy x) => x < (fProxy)0 ? (fProxy)(-1) : (fProxy)1;

        // sqrt(a^2 + b^2) without destructive underflow/overflow (NR pythag).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy pythag(fProxy a, fProxy b)
        {
            fProxy aa = math.abs(a), ab = math.abs(b);
            if (aa > ab) { fProxy r = ab / aa; return aa * math.sqrt((fProxy)1 + r * r); }
            if (ab == (fProxy)0) return (fProxy)0;
            { fProxy r = aa / ab; return ab * math.sqrt((fProxy)1 + r * r); }
        }
    }
}
