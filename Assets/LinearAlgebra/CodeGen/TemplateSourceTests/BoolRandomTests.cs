using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Tests for the bool random-fill core (Rand). Hand-written singular file (Rand
// is not generated per-type), mirroring the Chunk-1 RandomTests.fProxy.cs structure:
//   * nextBernoulliInPlace(ref rng, ref dest, p): element = rng.NextFloat() < p. Validates p in [0,1].
//   * nextBoolInPlace(ref rng, ref dest): fair coin via rng.NextBool().
// Burst-compatible computational tests live in TestJob (message-free asserts + Fail-buffer
// diagnostics); managed-throw guards are plain [Test] methods on the main thread. FIXED seeds.
public class BoolRandomTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            BernoulliP0AllFalse,
            BernoulliP1AllTrue,
            BernoulliHalfFraction,
            BernoulliDeterminism,
            NextBoolMix,
            MatrixOverloads,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<int> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.BernoulliP0AllFalse:    BernoulliP0AllFalse();    break;
                case TestType.BernoulliP1AllTrue:     BernoulliP1AllTrue();     break;
                case TestType.BernoulliHalfFraction:  BernoulliHalfFraction();  break;
                case TestType.BernoulliDeterminism:   BernoulliDeterminism();   break;
                case TestType.NextBoolMix:            NextBoolMix();            break;
                case TestType.MatrixOverloads:        MatrixOverloads();        break;
            }
        }

        const int N = 8192;

        // p = 0 => every element false.
        void BernoulliP0AllFalse()
        {
            var rng = new Random(1234567u);
            var v = new boolN(N, Allocator.Persistent);
            for (int i = 0; i < v.N; i++) v[i] = true;   // poison true
            Rand.nextBernoulliInPlace(ref rng, ref v, 0f);
            for (int i = 0; i < v.N; i++)
                AssertTrue(v[i] == false);
            v.Dispose();
        }

        // p = 1 => every element true.
        void BernoulliP1AllTrue()
        {
            var rng = new Random(2468013u);
            var v = new boolN(N, Allocator.Persistent);
            for (int i = 0; i < v.N; i++) v[i] = false;  // poison false
            Rand.nextBernoulliInPlace(ref rng, ref v, 1f);
            for (int i = 0; i < v.N; i++)
                AssertTrue(v[i] == true);
            v.Dispose();
        }

        // p = 0.5 over a large fill: empirical true-fraction ~ 0.5 within a loose tolerance.
        void BernoulliHalfFraction()
        {
            var rng = new Random(13572468u);
            var v = new boolN(N, Allocator.Persistent);
            Rand.nextBernoulliInPlace(ref rng, ref v, 0.5f);

            int trues = 0;
            for (int i = 0; i < v.N; i++)
                if (v[i]) trues++;
            float frac = (float)trues / v.N;
            AssertClose(frac, 0.5f, 0.03f);
            v.Dispose();
        }

        // Same seed + same p => identical buffer.
        void BernoulliDeterminism()
        {
            var r1 = new Random(99u);
            var v1 = new boolN(512, Allocator.Temp);
            Rand.nextBernoulliInPlace(ref r1, ref v1, 0.3f);

            var r2 = new Random(99u);
            var v2 = new boolN(512, Allocator.Temp);
            Rand.nextBernoulliInPlace(ref r2, ref v2, 0.3f);

            for (int i = 0; i < v1.N; i++)
                AssertTrue(v1[i] == v2[i]);
        }

        // nextBoolInPlace (fair coin) produces a mix: both true and false present.
        void NextBoolMix()
        {
            var rng = new Random(97531864u);
            var v = new boolN(N, Allocator.Persistent);
            Rand.nextBoolInPlace(ref rng, ref v);

            int trues = 0;
            for (int i = 0; i < v.N; i++)
                if (v[i]) trues++;
            AssertTrue(trues > 0);
            AssertTrue(trues < v.N);

            // and roughly fair
            float frac = (float)trues / v.N;
            AssertClose(frac, 0.5f, 0.03f);
            v.Dispose();
        }

        // Matrix overloads: Bernoulli(p=1) all true; nextBool produces a mix; all M*N written.
        void MatrixOverloads()
        {
            var rng = new Random(20240626u);

            var M = new boolMxN(8, 16, Allocator.Temp);
            for (int i = 0; i < M.Length; i++) M[i] = false;
            Rand.nextBernoulliInPlace(ref rng, ref M, 1f);
            AssertTrue(M.Length == 128);
            for (int i = 0; i < M.Length; i++)
                AssertTrue(M[i] == true);

            var Mb = new boolMxN(8, 16, Allocator.Temp);
            Rand.nextBoolInPlace(ref rng, ref Mb);
            int trues = 0;
            for (int i = 0; i < Mb.Length; i++)
                if (Mb[i]) trues++;
            AssertTrue(trues > 0 && trues < Mb.Length);
        }

        // ---------------- helpers ----------------

        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == 0)
            {
                Fail[0] = 1;
                Fail[1] = (int)(a * 10000f);
                Fail[2] = (int)(b * 10000f);
                Fail[3] = (int)(diff * 10000f);
            }
            Assert.IsTrue(diff <= precision);
        }

        void AssertTrue(bool ok)
        {
            if (!ok && Fail[0] == 0)
            {
                Fail[0] = 1;
                Fail[1] = -1;
                Fail[2] = -1;
                Fail[3] = -1;
            }
            Assert.IsTrue(ok);
        }
    }

    void RunJob(TestJob.TestType type)
    {
        var fail = new NativeArray<int>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != 0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} (x1e-4 for fractions)");
        }
        catch (Exception e)
        {
            if (fail[0] != 0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    [Test] public void BernoulliP0AllFalseTest() => RunJob(TestJob.TestType.BernoulliP0AllFalse);
    [Test] public void BernoulliP1AllTrueTest() => RunJob(TestJob.TestType.BernoulliP1AllTrue);
    [Test] public void BernoulliHalfFractionTest() => RunJob(TestJob.TestType.BernoulliHalfFraction);
    [Test] public void BernoulliDeterminismTest() => RunJob(TestJob.TestType.BernoulliDeterminism);
    [Test] public void NextBoolMixTest() => RunJob(TestJob.TestType.NextBoolMix);
    [Test] public void MatrixOverloadsTest() => RunJob(TestJob.TestType.MatrixOverloads);

    // ---------------- Managed validation throws (main thread, not in a Burst job) ----------------

    [Test]
    public void BernoulliPOutOfRangeThrows()
    {
        Random rng = new Random(1u);

        var v = new boolN(8, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Rand.nextBernoulliInPlace(ref rng, ref v, -0.01f));
        Assert.Throws<ArgumentException>(() => Rand.nextBernoulliInPlace(ref rng, ref v, 1.01f));

        var M = new boolMxN(3, 3, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => Rand.nextBernoulliInPlace(ref rng, ref M, -1f));
        Assert.Throws<ArgumentException>(() => Rand.nextBernoulliInPlace(ref rng, ref M, 2f));
    }
}
