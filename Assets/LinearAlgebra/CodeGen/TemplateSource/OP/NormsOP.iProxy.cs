using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace LinearAlgebra
{
    // Public Norms surface for the SIGNED integer family (int/short/long; uint is deliberately
    // excluded). Merges into the SAME bare partial class as NormsOP.fProxy.cs's `Norms`,
    // forwarding to the distinct intNormsCore/shortNormsCore/longNormsCore generic bodies below
    // (kept split so per-type generated copies can't collide as duplicate members, CS0111 --
    // same reason floatNormsCore/doubleNormsCore stay split).
    //
    // RETURN-TYPE WIDENING (locked convention): L1/LInf -> long (a sum/max of |x|, widened so it
    // can't overflow the source integer type); L2 -> double (a Euclidean length is generally
    // irrational even over integer input, e.g. L2({1,1}) == sqrt(2)).
    public static partial class Norms {

        // Standard L1 norm: the sum of absolute values, Σ|xᵢ| (NOT averaged by length), widened
        // to `long`. See iProxyNormsCore.L1 for the abs-overflow care this takes.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long L1(in iProxyN   a) => iProxyNormsCore.L1(a);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long L1(in iProxyMxN a) => iProxyNormsCore.L1(a);

        // L-infinity (max-abs) norm: the largest absolute element, max_i |xᵢ|, widened to `long`.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long LInf(in iProxyN   a) => iProxyNormsCore.LInf(a);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long LInf(in iProxyMxN a) => iProxyNormsCore.LInf(a);

        // Euclidean (L2) norm: sqrt(Σxᵢ²), as a `double` (see iProxyNormsCore.L2 for the
        // accumulation-precision note).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double L2(in iProxyN   a) => iProxyNormsCore.L2(a);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double L2(in iProxyMxN a) => iProxyNormsCore.L2(a);
    }

    // Internal generic bodies (one per norm kernel), one distinct type per generated type
    // (intNormsCore/shortNormsCore/longNormsCore) -- see the merge-safety note on `Norms` above.
    internal static class iProxyNormsCore
    {
        // Σ|xᵢ|, accumulated in `long`. Each element is widened to long BEFORE taking its
        // absolute value. For the two NARROWER signed types (int, short), this widening makes
        // the negation inside abs() exact: casting up to long first means negating their own
        // MinValue no longer overflows (e.g. L1({int.MinValue}) == 2147483648L, the true
        // magnitude, not a wrapped-negative garbage value). This widening does NOT help when the
        // generated type IS `long` itself: long is already the widest integer type available here
        // (no Int128), so casting long.MinValue to long is a no-op and math.abs(long.MinValue)
        // still overflows/wraps back to long.MinValue (a negative "absolute value") -- a
        // documented limitation, not fixed.
        //
        // MIXED-INPUT behavior for the long variant specifically (a LONE long.MinValue element is
        // the case above; this is what happens when OTHER elements are ALSO present): the wrapped
        // negative value simply participates in the running sum like any other term -- e.g.
        // L1({long.MinValue, 5}) == long.MinValue + 5 (a valid, in-range long addition, NOT a
        // further overflow). Contrast with LInf below, which -- for the exact same mixed input --
        // silently DROPS the long.MinValue element from consideration entirely (see LInf's doc).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long L1<T>(in T a) where T : unmanaged, IUnsafeiProxyArray
        {
            long sum = 0;
            for (int i = 0; i < a.Data.Length; i++)
                sum += math.abs((long)a.Data[i]);
            return sum;
        }

        // max_i |xᵢ|, in `long`. Same widen-before-abs care as L1. Seeded at `long.MinValue`
        // (NOT 0): for the `long` proxy type, abs(long.MinValue) itself wraps back to
        // long.MinValue (a negative "absolute value" -- the same documented not-fixable
        // limitation as L1). Seeding at 0 would silently swallow that wrapped value (0 > a
        // negative number is false, so max would never update away from 0 -- a plausible-
        // looking but WRONG answer that hides a huge-magnitude element as if it were zero).
        // Seeding at long.MinValue instead makes the wrap surface as long.MinValue, matching
        // L1's behavior for the same input, at zero cost to any legitimate input (every real
        // abs value is >= 0 > long.MinValue, so the comparison still always updates correctly).
        //
        // IMPORTANT ASYMMETRY vs L1 for MIXED inputs (long variant only): the fix above only
        // covers a LONE long.MinValue element. If the input ALSO contains any OTHER element,
        // that other element's non-negative abs value will always compare greater than the
        // wrapped-negative abs(long.MinValue), so LInf silently DROPS the long.MinValue element
        // from consideration entirely -- e.g. LInf({long.MinValue, 5}) == 5, NOT long.MinValue.
        // This is a MORE severe instance of the same not-fixable wraparound limitation: L1 still
        // reflects the MinValue element's contribution for mixed inputs (via wrap-summing, see
        // above), while LInf can silently ignore it whenever a non-pathological element is also
        // present. Not fixable without a wider type; both behaviors are pinned by a test.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long LInf<T>(in T a) where T : unmanaged, IUnsafeiProxyArray
        {
            // Empty input returns 0 gracefully (matching L1/L2's empty behavior) -- checked via
            // Data.Length directly, NOT by checking whether `max` is still at its long.MinValue
            // seed after the loop, because a genuinely non-empty {long.MinValue} input ALSO
            // leaves `max` at that same seed value (see below) -- a post-loop value-based check
            // would wrongly collapse that legitimate case back to 0, undoing the whole fix.
            if (a.Data.Length == 0) return 0;

            long max = long.MinValue;
            for (int i = 0; i < a.Data.Length; i++)
            {
                long abs = math.abs((long)a.Data[i]);
                if (abs > max) max = abs;
            }
            return max;
        }

        // sqrt(Σxᵢ²), accumulated in `double`. Squares are accumulated in double rather than
        // long: a long accumulator would itself overflow for a long enough vector of large int
        // values (Σ of many int.MaxValue² terms exceeds long.MaxValue), whereas double's ~15-17
        // significant decimal digits exactly cover int/short squares and approximately cover
        // long squares (the same precision tradeoff already documented on Stats.variance).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double L2<T>(in T a) where T : unmanaged, IUnsafeiProxyArray
        {
            double sum = 0.0;
            for (int i = 0; i < a.Data.Length; i++)
            {
                double v = (double)a.Data[i];
                sum += v * v;
            }
            return math.sqrt(sum);
        }
    }
}
