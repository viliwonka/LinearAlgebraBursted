using System.Runtime.CompilerServices;

using Unity.Burst;
using Unity.Mathematics;

//alsoExpand[uint]// bit-manipulation intrinsics mirroring Unity.Mathematics' math.* bit ops
//(countbits/tzcnt/lzcnt/reversebits/ror/rol/ceilpow2). These act on the BIT PATTERN of an element,
//not its numeric value, so - unlike UnsafeMathOP.iProxy.cs's abs/relu - they are fully sign-agnostic
//and need no skipFor exclusion for uint.
//
//short needs special handling throughout: Unity.Mathematics defines NO short overload for any of
//these ops (int/uint/long all have direct native overloads in com.unity.mathematics's math.cs -
//verified by inspection). Each op below therefore defines short's behaviour via an explicit 16-bit
//width correction over a zero-extended (uint)(ushort) reinterpretation of the value - see each op's
//own comment for the specific correction, and note where a correction would normally use bitwise-OR
//to combine two shifted halves, addition is used instead purely because the codegen //+choose[...]
//marker's OWN value-list separator is the '|' character - a literal '|' inside a choose branch would
//be misparsed as an extra branch (do not write that marker's literal token in prose here - the
//codegen parser is content-sensitive, not comment-aware).

namespace BULA.Internal
{
    public static unsafe partial class UnsafeBitsOP
    {
        // countbits (population count / Hamming weight). short is corrected by counting bits over
        // only the 16-bit zero-extended pattern - the zero-extension's top 16 bits are always 0 and
        // contribute nothing to the count, so the result range is 0..16 (vs 0..32 for int/uint and
        // 0..64 for long).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void countbits([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
            {
                iProxy v = x[i];
                x[i] = (iProxy)(/*+choose[math.countbits(v)|math.countbits((uint)(ushort)v)|math.countbits(v)|math.countbits(v)]*/math.countbits(v)/*-choose*/);
            }
        }

        // tzcnt (trailing zero count). short reinterprets the same zero-extended way, but tzcnt(0)
        // over the 32-bit zero-extended value returns 32 (int/uint's own convention), so it is
        // clamped down to 16 - short's actual bit width - via math.min. Every nonzero case already
        // lands correctly since the lowest set bit can only ever be within the low 16 bits.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void tzcnt([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
            {
                iProxy v = x[i];
                x[i] = (iProxy)(/*+choose[math.tzcnt(v)|math.min(16, math.tzcnt((uint)(ushort)v))|math.tzcnt(v)|math.tzcnt(v)]*/math.tzcnt(v)/*-choose*/);
            }
        }

        // lzcnt (leading zero count). The zero-extended 32-bit value always carries exactly 16 extra
        // leading zeros above the real 16-bit content (from the zero-extension itself), so
        // subtracting a flat 16 converts the 32-bit-width answer into the 16-bit-width one. This
        // holds for every input including 0 (32 - 16 == 16, short's own bit width - matching how
        // int/uint/long's own lzcnt(0) equals their bit width too).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void lzcnt([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
            {
                iProxy v = x[i];
                x[i] = (iProxy)(/*+choose[math.lzcnt(v)|math.lzcnt((uint)(ushort)v) - 16|math.lzcnt(v)|math.lzcnt(v)]*/math.lzcnt(v)/*-choose*/);
            }
        }

        // reversebits: reverse the bit pattern within the type's own width. Reversing the full
        // 32-bit zero-extended value puts the (reversed) real 16-bit content into the UPPER 16 bits
        // - since the original's upper 16 bits were all zero, and a full reversal swaps the hi/lo
        // halves - and zero into the lower 16; shifting right by 16 brings the correct 16-bit-reversed
        // pattern down into place.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void reversebits([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
            {
                iProxy v = x[i];
                x[i] = (iProxy)(/*+choose[math.reversebits(v)|math.reversebits((uint)(ushort)v) >> 16|math.reversebits(v)|math.reversebits(v)]*/math.reversebits(v)/*-choose*/);
            }
        }

        // rol/ror: Unity.Mathematics' own uint/ulong rol/ror rely on C#'s shift-count masking
        // (mod 32 / mod 64) to make the shift==0 edge case fall out for free AT THE CONTAINER WIDTH -
        // that trick doesn't hold here since short's 16-bit rotate is computed inside a wider 32-bit
        // uint container, so the rotate amount is normalized into [0,16) explicitly, once, ahead of
        // the loop (meaningful only for short - skipFor'd away for every other generated type, since
        // int/long/uint forward straight to math.rol/math.ror and let Unity.Mathematics own whatever
        // shift-amount behaviour it defines).
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void rol([NoAlias] iProxy* x, int n, int shift)
        {
            //+skipFor[int,long,uint]
            int shiftMod16 = ((shift % 16) + 16) % 16;
            //-skipFor

            for (int i = 0; i < n; i++)
            {
                iProxy v = x[i];
                x[i] = (iProxy)(/*+choose[math.rol(v, shift)|((uint)(ushort)v << shiftMod16) + ((uint)(ushort)v >> (16 - shiftMod16)) & 0xFFFFu|math.rol(v, shift)|math.rol(v, shift)]*/math.rol(v, shift)/*-choose*/);
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void ror([NoAlias] iProxy* x, int n, int shift)
        {
            //+skipFor[int,long,uint]
            int shiftMod16 = ((shift % 16) + 16) % 16;
            //-skipFor

            for (int i = 0; i < n; i++)
            {
                iProxy v = x[i];
                x[i] = (iProxy)(/*+choose[math.ror(v, shift)|((uint)(ushort)v >> shiftMod16) + ((uint)(ushort)v << (16 - shiftMod16)) & 0xFFFFu|math.ror(v, shift)|math.ror(v, shift)]*/math.ror(v, shift)/*-choose*/);
            }
        }

        // ceilpow2: smallest power of two >= v (Unity.Mathematics' own quirk - ceilpow2(0) == 0 - is
        // preserved for short too, see below). short's version avoids the OR-cascade Unity's own
        // int/uint/long ceilpow2 use internally (again: no literal '|' allowed inside a choose
        // branch) by going through the already-derived 16-bit lzcnt formula instead: for x > 0, the
        // smallest power of two >= x is `1 << (16 - lzcnt16(x - 1))`. This also happens to naturally
        // fall out to 0 for x == 0 (computes 1 << 16, which truncates to 0 in a 16-bit result) and to
        // 1 for x == 1, with no extra special-casing needed.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void ceilpow2([NoAlias] iProxy* x, int n)
        {
            for (int i = 0; i < n; i++)
            {
                iProxy v = x[i];
                x[i] = (iProxy)(/*+choose[math.ceilpow2(v)|1u << (32 - math.lzcnt((uint)(ushort)(v - 1)))|math.ceilpow2(v)|math.ceilpow2(v)]*/math.ceilpow2(v)/*-choose*/);
            }
        }
    }
}
