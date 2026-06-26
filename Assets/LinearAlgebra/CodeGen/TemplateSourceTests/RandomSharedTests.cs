using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;
using Random = Unity.Mathematics.Random;

// Tests for the type-agnostic shuffle / permutation / sampling ops in RandomOP (shared across all
// element types; not generated per-type). Hand-written singular file.
//   * randomPermutationInpl(ref Pivot, ref rng) : uniform permutation of 0..N-1 from identity;
//     Pivot.Sign reflects swap parity.
//   * shuffleInpl(ref Indices, ref rng)         : Fisher-Yates over existing contents.
//   * sampleKWithoutReplacementInpl(ref Indices dest, int n, ref rng) : dest.N distinct indices
//     from [0, n). Throws if n <= 0 or dest.N > n; dest.N == 0 returns without alloc/throw.
//
// Burst-compatible computational tests live in TestJob; managed-throw guards are plain [Test]
// methods on the main thread. FIXED seeds only.
public class RandomSharedTests
{
    [BurstCompile]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            PermutationIsBijection,
            PermutationN1Identity,
            PermutationSignMatchesParity,
            PermutationNonIdentity,
            ShuffleMultisetEqual,
            ShuffleN1Unchanged,
            SampleKDistinct,
            SampleKFullPermutation,
            SampleKZeroNoThrow,
            SampleKDeterminism,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected, [3] extra
        public NativeArray<int> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.PermutationIsBijection:       PermutationIsBijection();       break;
                case TestType.PermutationN1Identity:        PermutationN1Identity();        break;
                case TestType.PermutationSignMatchesParity: PermutationSignMatchesParity(); break;
                case TestType.PermutationNonIdentity:       PermutationNonIdentity();       break;
                case TestType.ShuffleMultisetEqual:         ShuffleMultisetEqual();         break;
                case TestType.ShuffleN1Unchanged:           ShuffleN1Unchanged();           break;
                case TestType.SampleKDistinct:              SampleKDistinct();              break;
                case TestType.SampleKFullPermutation:       SampleKFullPermutation();       break;
                case TestType.SampleKZeroNoThrow:           SampleKZeroNoThrow();           break;
                case TestType.SampleKDeterminism:           SampleKDeterminism();           break;
            }
        }

        // randomPermutationInpl yields a bijection of 0..N-1, and a Copy preserves contents+Sign.
        void PermutationIsBijection()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 16;
            var p = arena.Pivot(n);
            var rng = new Random(1234567u);
            RandomOP.randomPermutationInpl(ref p, ref rng);

            AssertTrue(p.N == n);
            // Each value in 0..n-1 appears exactly once.
            var seen = arena.Indices(n);
            for (int i = 0; i < n; i++) seen[i] = 0;
            for (int i = 0; i < n; i++)
            {
                int val = p[i];
                AssertTrue(val >= 0 && val < n);
                seen[val] = seen[val] + 1;
            }
            for (int i = 0; i < n; i++)
                AssertTrue(seen[i] == 1);

            AssertTrue(p.Sign == 1 || p.Sign == -1);

            var copy = p.Copy();
            AssertTrue(copy.Sign == p.Sign);
            for (int i = 0; i < n; i++)
                AssertTrue(copy[i] == p[i]);
            copy.Dispose();

            arena.Dispose();
        }

        // N == 1 => identity, even parity.
        void PermutationN1Identity()
        {
            var arena = new Arena(Allocator.Persistent);
            var p = arena.Pivot(1);
            var rng = new Random(99u);
            RandomOP.randomPermutationInpl(ref p, ref rng);
            AssertTrue(p[0] == 0);
            AssertTrue(p.Sign == 1);
            arena.Dispose();
        }

        // Pivot.Sign parity must equal the inversion-count parity of the resulting permutation.
        // (No managed arrays: Burst forbids them, so the (n, seed) grid is walked arithmetically.)
        void PermutationSignMatchesParity()
        {
            var arena = new Arena(Allocator.Persistent);
            for (int n = 2; n <= 13; n++)
            {
                for (uint k = 0; k < 6; k++)
                {
                    uint seed = (uint)(n * 131) + k * 2654435761u + 1u;
                    CheckSignParity(ref arena, n, seed);
                }
            }
            arena.Dispose();
        }

        void CheckSignParity(ref Arena arena, int n, uint seed)
        {
            var p = arena.Pivot(n);
            var rng = new Random(seed);
            RandomOP.randomPermutationInpl(ref p, ref rng);

            int inversions = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    if (p[i] > p[j]) inversions++;

            int invOdd = inversions & 1;
            int signOdd = (p.Sign == -1) ? 1 : 0;
            AssertEq(signOdd, invOdd);
        }

        // Over a few seeds, at least one permutation of 0..N-1 is non-identity.
        void PermutationNonIdentity()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;
            bool anyNonIdentity = false;
            for (uint s = 1; s <= 8; s++)
            {
                var p = arena.Pivot(n);
                var rng = new Random(s * 2654435761u + 1u);
                RandomOP.randomPermutationInpl(ref p, ref rng);
                for (int i = 0; i < n; i++)
                    if (p[i] != i) { anyNonIdentity = true; break; }
            }
            AssertTrue(anyNonIdentity);
            arena.Dispose();
        }

        // shuffleInpl keeps the multiset of contents (incl. duplicates), only reordering.
        void ShuffleMultisetEqual()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;

            // Known multiset with duplicates (set element-wise; Burst forbids managed arrays).
            var idx = arena.Indices(n);
            var pre = arena.Indices(n);
            idx[0] = 3; idx[1] = 3; idx[2] = 1; idx[3] = 7;
            idx[4] = 7; idx[5] = 2; idx[6] = 9; idx[7] = 0;
            for (int i = 0; i < n; i++) pre[i] = idx[i];

            var rng = new Random(2468013u);
            RandomOP.shuffleInpl(ref idx, ref rng);

            // sort both (insertion sort) and compare element-wise => multiset equality
            InsertionSort(ref pre);
            InsertionSort(ref idx);
            for (int i = 0; i < n; i++)
                AssertEq(idx[i], pre[i]);

            arena.Dispose();
        }

        // N == 1 shuffle leaves the single element unchanged.
        void ShuffleN1Unchanged()
        {
            var arena = new Arena(Allocator.Persistent);
            var idx = arena.Indices(1);
            idx[0] = 42;
            var rng = new Random(7u);
            RandomOP.shuffleInpl(ref idx, ref rng);
            AssertEq(idx[0], 42);
            arena.Dispose();
        }

        // sampleK: dest.N distinct indices in [0, n).
        void SampleKDistinct()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 20, k = 5;
            var dest = arena.Indices(k);
            var rng = new Random(13572468u);
            RandomOP.sampleKWithoutReplacementInpl(ref dest, n, ref rng);

            for (int i = 0; i < k; i++)
            {
                AssertTrue(dest[i] >= 0 && dest[i] < n);
                for (int j = i + 1; j < k; j++)
                    AssertTrue(dest[i] != dest[j]);   // distinct
            }
            arena.Dispose();
        }

        // k == n => a full permutation of 0..n-1.
        void SampleKFullPermutation()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 8;
            var dest = arena.Indices(n);
            var rng = new Random(97531864u);
            RandomOP.sampleKWithoutReplacementInpl(ref dest, n, ref rng);

            var seen = arena.Indices(n);
            for (int i = 0; i < n; i++) seen[i] = 0;
            for (int i = 0; i < n; i++)
            {
                AssertTrue(dest[i] >= 0 && dest[i] < n);
                seen[dest[i]] = seen[dest[i]] + 1;
            }
            for (int i = 0; i < n; i++)
                AssertEq(seen[i], 1);
            arena.Dispose();
        }

        // dest.N == 0 returns without alloc/throw.
        void SampleKZeroNoThrow()
        {
            var arena = new Arena(Allocator.Persistent);
            var dest = arena.Indices(0);
            var rng = new Random(1u);
            RandomOP.sampleKWithoutReplacementInpl(ref dest, 5, ref rng);
            AssertEq(dest.N, 0);
            arena.Dispose();
        }

        // Fixed seed => identical sample.
        void SampleKDeterminism()
        {
            var arena = new Arena(Allocator.Persistent);
            int n = 50, k = 10;
            var d1 = arena.Indices(k);
            var r1 = new Random(555u);
            RandomOP.sampleKWithoutReplacementInpl(ref d1, n, ref r1);

            var d2 = arena.Indices(k);
            var r2 = new Random(555u);
            RandomOP.sampleKWithoutReplacementInpl(ref d2, n, ref r2);

            for (int i = 0; i < k; i++)
                AssertEq(d1[i], d2[i]);
            arena.Dispose();
        }

        // ---------------- helpers ----------------

        void InsertionSort(ref Indices a)
        {
            int n = a.N;
            for (int i = 1; i < n; i++)
            {
                int key = a[i];
                int j = i - 1;
                while (j >= 0 && a[j] > key)
                {
                    a[j + 1] = a[j];
                    j--;
                }
                a[j + 1] = key;
            }
        }

        void AssertEq(int got, int expected)
        {
            if (got != expected && Fail[0] == 0)
            {
                Fail[0] = 1;
                Fail[1] = got;
                Fail[2] = expected;
                Fail[3] = 0;
            }
            Assert.AreEqual(expected, got);
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
                Assert.Fail($"got {fail[1]}, expected {fail[2]}, extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != 0)
                Assert.Fail($"{type}: got {fail[1]}, expected {fail[2]}, extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    [Test] public void PermutationIsBijectionTest() => RunJob(TestJob.TestType.PermutationIsBijection);
    [Test] public void PermutationN1IdentityTest() => RunJob(TestJob.TestType.PermutationN1Identity);
    [Test] public void PermutationSignMatchesParityTest() => RunJob(TestJob.TestType.PermutationSignMatchesParity);
    [Test] public void PermutationNonIdentityTest() => RunJob(TestJob.TestType.PermutationNonIdentity);
    [Test] public void ShuffleMultisetEqualTest() => RunJob(TestJob.TestType.ShuffleMultisetEqual);
    [Test] public void ShuffleN1UnchangedTest() => RunJob(TestJob.TestType.ShuffleN1Unchanged);
    [Test] public void SampleKDistinctTest() => RunJob(TestJob.TestType.SampleKDistinct);
    [Test] public void SampleKFullPermutationTest() => RunJob(TestJob.TestType.SampleKFullPermutation);
    [Test] public void SampleKZeroNoThrowTest() => RunJob(TestJob.TestType.SampleKZeroNoThrow);
    [Test] public void SampleKDeterminismTest() => RunJob(TestJob.TestType.SampleKDeterminism);

    // ---------------- Managed validation throws (main thread, not in a Burst job) ----------------

    [Test]
    public void SampleKInvalidArgsThrow()
    {
        var arena = new Arena(Allocator.Persistent);
        Random rng = new Random(1u);

        // n <= 0 throws.
        var dest = arena.Indices(3);
        Assert.Throws<ArgumentException>(() => RandomOP.sampleKWithoutReplacementInpl(ref dest, 0, ref rng));
        Assert.Throws<ArgumentException>(() => RandomOP.sampleKWithoutReplacementInpl(ref dest, -1, ref rng));

        // dest.N > n throws.
        var big = arena.Indices(5);
        Assert.Throws<ArgumentException>(() => RandomOP.sampleKWithoutReplacementInpl(ref big, 3, ref rng));

        arena.Dispose();
    }
}
