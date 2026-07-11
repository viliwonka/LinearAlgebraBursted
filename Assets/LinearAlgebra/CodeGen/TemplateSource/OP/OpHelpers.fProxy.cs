using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Scalar helpers shared by the Householder/QL/bidiagonal kernels (QR, LQ, Bidiag, SVD, Eigen).
    internal static class fProxyOpHelpers
    {
        // |a| with the sign of b (Fortran SIGN / C99 copysign, NR convention): b >= 0 -> +|a|,
        // INCLUDING b == 0. Deliberately not math.sign-based: math.sign(0) == 0, which would zero
        // out the Householder sign choice (copysign(1, 0) must be +1 so a reflector built on a
        // zero pivot entry still gets a nonzero shift).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static fProxy copysign(fProxy a, fProxy b) => b >= (fProxy)0 ? math.abs(a) : -math.abs(a);

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
