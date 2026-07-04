using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for the integer Norms surface (int / short / long): L1 (Sigma|x|, widened to long),
// LInf (max|x|, widened to long), L2 (sqrt(Sigma x^2), as double). None of these throw on empty --
// the loop simply does not execute and they return 0 / 0.0. Oracles are exact where the result type
// is exact (L1/LInf -> long, and the Pythagorean-triple L2 cases -> exact double) and use a tiny
// tolerance only for the irrational sqrt(2) case.
//
// Several abs-overflow edge cases are pinned per generated type via the //+choose marker (see the
// MinValueAbsOverflow and MinValueMixedInput cases for the full reasoning) -- expected values
// genuinely differ per type, not just in magnitude. MinValueMixedInput additionally pins a
// long-only ASYMMETRY between L1 (wrap-sums a pathological MinValue element) and LInf (silently
// DROPS it once any other element is present).
public class longNormsTests
{
    [BurstCompile]
    public struct NormsTestJob : IJob
    {
        public enum TestType
        {
            BasicVector,
            PythagoreanTriple,
            AllNegative,
            Matrix,
            AllZero,
            Empty,
            MinValueAbsOverflow,
            MinValueMixedInput,
        }

        public TestType Type;

        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            try
            {
                switch (Type)
                {
                    case TestType.BasicVector: BasicVector(ref arena); break;
                    case TestType.PythagoreanTriple: PythagoreanTriple(ref arena); break;
                    case TestType.AllNegative: AllNegative(ref arena); break;
                    case TestType.Matrix: Matrix(ref arena); break;
                    case TestType.AllZero: AllZero(ref arena); break;
                    case TestType.Empty: Empty(ref arena); break;
                    case TestType.MinValueAbsOverflow: MinValueAbsOverflow(ref arena); break;
                    case TestType.MinValueMixedInput: MinValueMixedInput(ref arena); break;
                    default: throw new NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        bool Close(double a, double b) => math.abs(a - b) < 1e-9;

        // {3,-4}: L1 = 3+4 = 7, LInf = 4, L2 = sqrt(9+16) = 5.
        void BasicVector(ref Arena arena)
        {
            var v = arena.longVec(2);
            v[0] = (long)3; v[1] = (long)(-4);
            Assert.IsTrue(Norms.L1(in v) == 7L);
            Assert.IsTrue(Norms.LInf(in v) == 4L);
            Assert.IsTrue(Close(Norms.L2(in v), 5.0));
        }

        // {1,2,2}: L1 = 5, LInf = 2, L2 = sqrt(1+4+4) = 3 (exact).
        void PythagoreanTriple(ref Arena arena)
        {
            var v = arena.longVec(3);
            v[0] = (long)1; v[1] = (long)2; v[2] = (long)2;
            Assert.IsTrue(Norms.L1(in v) == 5L);
            Assert.IsTrue(Norms.LInf(in v) == 2L);
            Assert.IsTrue(Close(Norms.L2(in v), 3.0));

            // Irrational case {1,1}: L2 = sqrt(2).
            var w = arena.longVec(2, (long)1);
            Assert.IsTrue(Close(Norms.L2(in w), math.sqrt(2.0)));
        }

        // Abs is applied: {-1,-2,-2} matches {1,2,2}: L1 = 5, LInf = 2, L2 = 3.
        void AllNegative(ref Arena arena)
        {
            var v = arena.longVec(3);
            v[0] = (long)(-1); v[1] = (long)(-2); v[2] = (long)(-2);
            Assert.IsTrue(Norms.L1(in v) == 5L);
            Assert.IsTrue(Norms.LInf(in v) == 2L);
            Assert.IsTrue(Close(Norms.L2(in v), 3.0));
        }

        // Matrix treated as one flat distribution {{3,-4},{0,0}} == {3,-4,0,0}: L1=7, LInf=4, L2=5.
        void Matrix(ref Arena arena)
        {
            var A = arena.longMat(2, 2, (long)0);
            A[0, 0] = (long)3; A[0, 1] = (long)(-4);
            Assert.IsTrue(Norms.L1(in A) == 7L);
            Assert.IsTrue(Norms.LInf(in A) == 4L);
            Assert.IsTrue(Close(Norms.L2(in A), 5.0));
        }

        // All-zero -> every norm 0.
        void AllZero(ref Arena arena)
        {
            var v = arena.longVec(4, (long)0);
            Assert.IsTrue(Norms.L1(in v) == 0L);
            Assert.IsTrue(Norms.LInf(in v) == 0L);
            Assert.IsTrue(Close(Norms.L2(in v), 0.0));
        }

        // Empty vector: norms return 0 / 0.0 GRACEFULLY (no throw -- the accumulation loop just
        // never runs). This is the documented contrast with Stats, which throws on empty.
        void Empty(ref Arena arena)
        {
            var v = arena.longVec(0);
            Assert.IsTrue(Norms.L1(in v) == 0L);
            Assert.IsTrue(Norms.LInf(in v) == 0L);
            Assert.IsTrue(Close(Norms.L2(in v), 0.0));
        }

        // ABS-OVERFLOW EDGE: single element == long.MinValue. The kernel widens each element to
        // long BEFORE taking abs, so for int/short the negation is exact and L1==LInf== the true
        // magnitude (int: 2147483648, short: 32768). For the LONG variant this widen-before-abs
        // trick cannot help -- long is already the widest integer type (no Int128), so
        // math.abs((long)long.MinValue) == max(-long.MinValue, long.MinValue) == long.MinValue
        // (a wrapped, still-NEGATIVE "absolute value"). This is a DOCUMENTED-LIMITATION behavior
        // that this test PINS (it is not fixable without a wider type):
        //   * L1   accumulates that negative value -> long.MinValue.
        //   * LInf seeds its running max at long.MinValue (NOT 0 -- an earlier 0-seeded version
        //     of this kernel silently swallowed the wrap and returned 0, a plausible-looking but
        //     WRONG answer; see NormsOP.long.cs's LInf doc comment), so it also surfaces ->
        //     long.MinValue, consistent with L1.
        // The mathematically-true magnitude 2^63 fits in NEITHER long branch, so we pin the actual
        // (broken-but-expected) outputs rather than an unrepresentable "correct" value.
        void MinValueAbsOverflow(ref Arena arena)
        {
            var v = arena.longVec(1);
            v[0] = (long)long.MinValue;

            long expectedL1 = long.MinValue;
            long expectedLInf = long.MinValue;

            Assert.IsTrue(Norms.L1(in v) == expectedL1);
            Assert.IsTrue(Norms.LInf(in v) == expectedLInf);
        }

        // MIXED-INPUT contrast (see NormsOP.long.cs's L1/LInf doc comments for the full
        // reasoning): a {MinValue, 5} vector. For int/short, both norms behave normally (MinValue
        // widens exactly, as in MinValueAbsOverflow above) -- L1 sums both magnitudes, LInf
        // correctly picks the larger one. For the LONG variant specifically, L1 and LInf DISAGREE
        // on how the pathological wrapped abs(long.MinValue) is handled:
        //   * L1   wrap-adds it into the running sum -> long.MinValue + 5 (a valid, in-range
        //     long addition -- still reflects the MinValue element's presence).
        //   * LInf silently DROPS it -> 5 (the legitimate abs(5)==5 beats the wrapped-negative
        //     "abs" of long.MinValue, so the huge-magnitude element is invisible to the result).
        // Both current (documented, not-fixable-without-a-wider-type) behaviors are pinned here.
        void MinValueMixedInput(ref Arena arena)
        {
            var v = arena.longVec(2);
            v[0] = (long)long.MinValue;
            v[1] = (long)5;

            long expectedL1 = long.MinValue + 5;
            long expectedLInf = 5L;

            Assert.IsTrue(Norms.L1(in v) == expectedL1);
            Assert.IsTrue(Norms.LInf(in v) == expectedLInf);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(NormsTestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void NormsCases(NormsTestJob.TestType type)
    {
        new NormsTestJob() { Type = type }.Run();
    }
}
