using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Collections;
using Random = Unity.Mathematics.Random;

// Concretely-long tests for the long-only out-of-int-range guard in longRandomOP.nextUniformInpl.
//
// WHY THIS FILE IS NOT A TEMPLATE: longRandomOP.nextUniformInpl draws via Random.NextInt (int
// bounds), so for the long expansion min/max must lie within [int.MinValue, int.MaxValue]; values
// outside throw. That guard is unreachable from the iProxy template: the proxy and the int/short
// expansions can never hold a value outside int range, and literals like (long)int.MinValue - 1
// do not even compile for the int/short type substitutions. The codegen has no per-type gate, and
// the firstpass proxy test assembly cannot reference the generated longRandomOP, so this test is
// hand-written directly in the real BurstLinearAlgebra.Tests assembly against the long expansion.
//
// Managed-only ([Test] on the main thread); no Burst job. FIXED seeds only.
public class RandomLongRangeTests
{
    [Test]
    public void MinBelowIntRangeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        Random rng = new Random(1u);

        long badMin = (long)int.MinValue - 1;   // just below int range

        var v = arena.longVec(8);
        Assert.Throws<ArgumentException>(() => longRandomOP.nextUniformInpl(ref rng, ref v, badMin, 0L));

        var M = arena.longMat(3, 3);
        Assert.Throws<ArgumentException>(() => longRandomOP.nextUniformInpl(ref rng, ref M, badMin, 0L));

        arena.Dispose();
    }

    [Test]
    public void MaxAboveIntRangeThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        Random rng = new Random(1u);

        long badMax = (long)int.MaxValue + 1;   // just above int range

        var v = arena.longVec(8);
        Assert.Throws<ArgumentException>(() => longRandomOP.nextUniformInpl(ref rng, ref v, 0L, badMax));

        var M = arena.longMat(3, 3);
        Assert.Throws<ArgumentException>(() => longRandomOP.nextUniformInpl(ref rng, ref M, 0L, badMax));

        arena.Dispose();
    }

    // Positive control: bounds within int range actually exercise the long fill path (no throw,
    // every element in range), proving the throws above are the guard firing — not a dead method.
    [Test]
    public void InIntRangeFillsAndDoesNotThrow()
    {
        var arena = new Arena(Allocator.Persistent);
        Random rng = new Random(20240626u);

        long min = -5L, max = 10L;
        var v = arena.longVec(4096);
        for (int i = 0; i < v.N; i++) v[i] = 999L;   // poison
        longRandomOP.nextUniformInpl(ref rng, ref v, min, max);
        for (int i = 0; i < v.N; i++)
            Assert.IsTrue(v[i] >= min && v[i] < max, $"v[{i}]={v[i]} out of [{min},{max})");

        // min == max constant-fill on the long path.
        var c = arena.longVec(32);
        longRandomOP.nextUniformInpl(ref rng, ref c, 7L, 7L);
        for (int i = 0; i < c.N; i++)
            Assert.AreEqual(7L, c[i]);

        arena.Dispose();
    }

    [Test]
    public void MinGreaterThanMaxStillThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        Random rng = new Random(1u);

        var v = arena.longVec(8);
        Assert.Throws<ArgumentException>(() => longRandomOP.nextUniformInpl(ref rng, ref v, 5L, 1L));

        arena.Dispose();
    }
}
