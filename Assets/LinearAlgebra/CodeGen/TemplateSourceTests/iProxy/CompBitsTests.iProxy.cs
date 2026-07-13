using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

//alsoExpand[uint]// bit ops (countbits/tzcnt/lzcnt/reversebits/ror/rol/ceilpow2) are sign-agnostic -
//they act on the bit pattern, not the numeric value - so this template covers uint too (contrast
//with CompMathTests.iProxy.cs, whose abs/relu genuinely have no unsigned meaning). Exact oracles
//only, no tolerance. SHORT is where width-correction bugs would hide (Unity.Mathematics defines no
//short overload for any of these ops - see UnsafeBitsOP.iProxy.cs) so its 16-bit-width semantics are
//exercised explicitly throughout (tzcnt/lzcnt/countbits of the type's OWN bit width, not 32).
public class iProxyCompBitsTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            BitPatterns,
            Reversebits,
            Ceilpow2,
            RolRorRoundTrip,
            RolRorKnownValues,
            ScalarShiftedByVector,
        }

        public TestType Type;

        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            try
            {
                switch (Type)
                {
                    case TestType.BitPatterns: BitPatternsTest(ref arena); break;
                    case TestType.Reversebits: ReversebitsTest(ref arena); break;
                    case TestType.Ceilpow2: Ceilpow2Test(ref arena); break;
                    case TestType.RolRorRoundTrip: RolRorRoundTripTest(ref arena); break;
                    case TestType.RolRorKnownValues: RolRorKnownValuesTest(ref arena); break;
                    case TestType.ScalarShiftedByVector: ScalarShiftedByVectorTest(ref arena); break;
                    default: throw new NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        // Bit width of the generated type (32 for int/uint, 16 for short, 64 for long) - drives every
        // "of 0 == width" / "of all-ones == width" oracle below without hand-duplicating the number
        // per type.
        private int Width => /*+choose[32|16|64|32]*/32/*-choose*/;

        // Bit pattern with ONLY the top (most significant) bit of the type's own width set - e.g.
        // int.MinValue for int, short.MinValue for short. This is the classic "sign bit" pattern,
        // which is exactly why it needs an explicit per-type literal/cast (int/short/long all go
        // negative; uint stays the large positive value 0x80000000u).
        private iProxy Msb => /*+choose[unchecked((int)0x80000000)|unchecked((short)0x8000)|unchecked((long)0x8000000000000000)|0x80000000u]*/unchecked((int)0x80000000)/*-choose*/;

        // Alternating 0b0101...01 bit pattern spanning the type's own full width (top bit always 0
        // by construction).
        private iProxy Alt => /*+choose[0x55555555|(short)0x5555|0x5555555555555555L|0x55555555u]*/0x55555555/*-choose*/;

        // All bits set within the type's own width (== -1 for every signed type; uint.MaxValue for
        // uint).
        private iProxy AllOnes => /*+choose[-1|(short)(-1)|-1L|0xFFFFFFFFu]*/-1/*-choose*/;

        private void BitPatternsTest(ref Arena arena)
        {
            int width = Width;
            iProxy msb = Msb;
            iProxy alt = Alt;
            iProxy allOnes = AllOnes;

            int n = 5;
            iProxyN v = arena.iProxyVec(n);
            v[0] = 0;
            v[1] = 1;
            v[2] = msb;
            v[3] = alt;
            v[4] = allOnes;

            // countbits: 0 -> 0, 1 -> 1, MSB-only -> 1, alternating -> width/2, all-ones -> width.
            iProxyN c = v.Copy();
            c.countbitsInPlace();
            Assert.IsTrue(c[0] == (iProxy)0);
            Assert.IsTrue(c[1] == (iProxy)1);
            Assert.IsTrue(c[2] == (iProxy)1);
            Assert.IsTrue(c[3] == (iProxy)(width / 2));
            Assert.IsTrue(c[4] == (iProxy)width);

            // tzcnt: 0 -> width (NOT 32 - the short case, tzcnt((short)0) == 16, is the whole point
            // of driving this off Width rather than a hardcoded 32), 1 -> 0, MSB-only -> width-1
            // (every bit below the top one is 0), all-ones -> 0 (bit0 is set).
            iProxyN t = v.Copy();
            t.tzcntInPlace();
            Assert.IsTrue(t[0] == (iProxy)width);
            Assert.IsTrue(t[1] == (iProxy)0);
            Assert.IsTrue(t[2] == (iProxy)(width - 1));
            Assert.IsTrue(t[3] == (iProxy)0); // alternating pattern's bit0 is set
            Assert.IsTrue(t[4] == (iProxy)0);

            // lzcnt: 0 -> width (short: lzcnt((short)0) == 16, not 32), 1 -> width-1, MSB-only -> 0
            // (the very top bit is set), alternating -> 1 (top bit is 0, the next one down is 1),
            // all-ones -> 0.
            iProxyN l = v.Copy();
            l.lzcntInPlace();
            Assert.IsTrue(l[0] == (iProxy)width);
            Assert.IsTrue(l[1] == (iProxy)(width - 1));
            Assert.IsTrue(l[2] == (iProxy)0);
            Assert.IsTrue(l[3] == (iProxy)1);
            Assert.IsTrue(l[4] == (iProxy)0);
        }

        private void ReversebitsTest(ref Arena arena)
        {
            iProxy msb = Msb;
            iProxy allOnes = AllOnes;

            int n = 5;
            iProxyN v = arena.iProxyVec(n);
            v[0] = 0;
            v[1] = 1;
            v[2] = msb;
            v[3] = Alt;
            v[4] = allOnes;

            iProxyN r = v.Copy();
            r.reversebitsInPlace();

            Assert.IsTrue(r[0] == (iProxy)0);   // reverse(0) == 0
            Assert.IsTrue(r[1] == msb);         // reverse(1) == MSB-only pattern
            Assert.IsTrue(r[2] == (iProxy)1);   // reverse(MSB-only) == 1
            Assert.IsTrue(r[4] == allOnes);     // reverse(all-ones) == all-ones (any width)

            // Round-trip: reversing twice restores the original - a self-consistent oracle for the
            // `alt` pattern (index 3) without needing to hand-compute its reversed literal per type.
            iProxyN r2 = r.Copy();
            r2.reversebitsInPlace();
            for (int i = 0; i < n; i++)
                Assert.IsTrue(r2[i] == v[i]);
        }

        private void Ceilpow2Test(ref Arena arena)
        {
            // Small values only - safe across int/short/long/uint with no overflow, and the short
            // width-corrected formula (see UnsafeBitsOP.iProxy.cs) is exercised the same as the
            // others since these all fit comfortably within 16 bits.
            int n = 11;
            iProxyN v = arena.iProxyVec(n);
            v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3; v[4] = 4; v[5] = 5;
            v[6] = 6; v[7] = 7; v[8] = 8; v[9] = 9; v[10] = 17;

            iProxyN c = v.Copy();
            c.ceilpow2InPlace();

            Assert.IsTrue(c[0] == (iProxy)0);   // Unity.Mathematics' own quirk: ceilpow2(0) == 0
            Assert.IsTrue(c[1] == (iProxy)1);
            Assert.IsTrue(c[2] == (iProxy)2);
            Assert.IsTrue(c[3] == (iProxy)4);
            Assert.IsTrue(c[4] == (iProxy)4);
            Assert.IsTrue(c[5] == (iProxy)8);
            Assert.IsTrue(c[6] == (iProxy)8);
            Assert.IsTrue(c[7] == (iProxy)8);
            Assert.IsTrue(c[8] == (iProxy)8);
            Assert.IsTrue(c[9] == (iProxy)16);
            Assert.IsTrue(c[10] == (iProxy)32);

            // High-value / wrap boundary cases (adversarial-review addition). 0x4000 (2^14) is
            // already a power of two for every generated type - ceilpow2 of an existing power of two
            // returns itself, with no wraparound concern since 0x4000 is tiny relative to every
            // type's own range. 0x4001's next power of two is 0x8000 (2^15 == 32768): still an
            // ordinary positive value for int/long/uint, but for short - whose positive range tops
            // out at 32767 - this genuinely overflows and wraps to short.MinValue (the sign bit
            // alone), hence the per-type expected literal below.
            int n2 = 2;
            iProxyN v2 = arena.iProxyVec(n2);
            v2[0] = 0x4000;
            v2[1] = 0x4001;

            iProxyN c2 = v2.Copy();
            c2.ceilpow2InPlace();

            Assert.IsTrue(c2[0] == (iProxy)0x4000);
            Assert.IsTrue(c2[1] == /*+choose[32768|unchecked((short)0x8000)|32768L|32768u]*/32768/*-choose*/);

            //+skipFor[uint]
            // Negative input has no meaning for uint - this sub-case is signed-types-only. Both
            // Unity.Mathematics' own ceilpow2 bit-trick (int/long) and this file's short-specific
            // lzcnt-based equivalent reduce ANY non-positive input down to 0, generalizing the
            // ceilpow2(0) == 0 quirk already exercised above. -7 is small and unremarkable - chosen
            // only to be unambiguously negative, not close to any type's own overflow boundary.
            iProxyN v3 = arena.iProxyVec(1);
            v3[0] = -7;
            iProxyN c3 = v3.Copy();
            c3.ceilpow2InPlace();
            Assert.IsTrue(c3[0] == (iProxy)0);
            //-skipFor
        }

        private void RolRorRoundTripTest(ref Arena arena)
        {
            int width = Width;

            int n = 4;
            iProxyN v = arena.iProxyVec(n);
            v[0] = 1;
            v[1] = Msb;
            v[2] = Alt;
            v[3] = AllOnes;

            // ror(n) undoing rol(n) (and vice versa is implied) must restore the original for any
            // in-range shift amount - exercised at 0, 1, a mid shift, width-1 (the largest valid
            // single-rotation amount for this type), and width itself (the modulo boundary where the
            // short-specific shiftMod16 normalization - and int/long/uint's native C# shift-count
            // masking - both collapse back down to a shift of 0, i.e. a full-circle rotation).
            RoundTripAt(v, 0);
            RoundTripAt(v, 1);
            RoundTripAt(v, 3);
            RoundTripAt(v, width - 1);
            RoundTripAt(v, width);
        }

        private void RoundTripAt(iProxyN v, int shift)
        {
            iProxyN rotated = v.Copy();
            rotated.rolInPlace(shift);
            rotated.rorInPlace(shift);
            for (int i = 0; i < v.N; i++)
                Assert.IsTrue(rotated[i] == v[i]);
        }

        private void RolRorKnownValuesTest(ref Arena arena)
        {
            iProxy msb = Msb;

            int n = 2;
            iProxyN v = arena.iProxyVec(n);
            v[0] = 1;
            v[1] = msb;

            iProxyN r = v.Copy();
            r.rolInPlace(1);
            Assert.IsTrue(r[0] == (iProxy)2); // rol(1, 1) == 2
            Assert.IsTrue(r[1] == (iProxy)1); // rol(MSB-only, 1) wraps around to 1

            iProxyN s = v.Copy();
            s.rorInPlace(1);
            Assert.IsTrue(s[0] == msb); // ror(1, 1) wraps around to the MSB-only pattern
        }

        private void ScalarShiftedByVectorTest(ref Arena arena)
        {
            // bitwiseLeftShiftInPlace(value, vec) computes vec[i] = value << vec[i] at the TYPE'S
            // OWN width. Width-2 is the regression case: for long that is a shift of 62, which a
            // 32-bit evaluation (count masked mod 32, result truncated) gets silently wrong.
            int width = Width;

            iProxyN v = arena.iProxyVec(2);
            v[0] = 1;
            v[1] = (iProxy)(width - 2);

            iProxyComp.bitwiseLeftShiftInPlace((iProxy)1, v);
            Assert.IsTrue(v[0] == (iProxy)2);   // 1 << 1
            // 1 << (width-2): top-bit-minus-one pattern of the type's own width.
            Assert.IsTrue(v[1] == /*+choose[0x40000000|(short)0x4000|0x4000000000000000L|0x40000000u]*/0x40000000/*-choose*/);

            // Right shift of the MSB-only pattern: arithmetic (sign-filling) for signed types,
            // logical for uint - both at the type's own width.
            iProxyN w = arena.iProxyVec(2);
            w[0] = 1;
            w[1] = (iProxy)(width - 2);

            iProxyComp.bitwiseRightShiftInPlace(Msb, w);
            Assert.IsTrue(w[0] == /*+choose[unchecked((int)0xC0000000)|unchecked((short)0xC000)|unchecked((long)0xC000000000000000)|0x40000000u]*/unchecked((int)0xC0000000)/*-choose*/);
            Assert.IsTrue(w[1] == /*+choose[-2|(short)(-2)|-2L|2u]*/-2/*-choose*/);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestsJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void Test(TestsJob.TestType type)
    {
        new TestsJob() { Type = type }.Run();
    }
}
