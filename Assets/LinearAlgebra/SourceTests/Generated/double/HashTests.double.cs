using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Property tests for the float/double Hash surface (Hash.hash / rowHashes / colHashes over
// doubleN / doubleMxN). These are pure PROPERTY checks (determinism, seed sensitivity, avalanche,
// length sensitivity, row/col consistency) that must hold for BOTH the float and the double
// expansion of this template -- no hard-coded hash constants live here (those are pinned once, in
// the managed, concrete-typed Assets/LinearAlgebra/SourceTests/HashSourceTests.cs, where they can
// be verified against the two independent reference implementations without Burst float folding).
//
// CODEGEN NOTE: every rowHashes/colHashes result is a uint buffer (uintN), a type that does NOT
// exist in TemplateSource's own standalone raw compile -- so it is only ever held in a `var` local
// (never the literal `uintN`), and both its elements and the expected uint are funneled through
// `(int)x` before comparison -- a single-step cast that compiles in the raw pass (dest[r] is the
// `iProxy` placeholder there, iProxy->int is a user-defined implicit conversion) AND in the
// generated code (dest[r] is `uint`, uint->int is a standard reinterpret). Both sides get the same
// reinterpret, so equality is preserved. A direct `(uint)dest[r]` does NOT compile in the raw pass
// (iProxy->uint is not a single legal conversion). Writing `uintN` literally would also fail the
// raw pass, exactly the trap the production code routes around with choose markers.
public class doubleHashTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct HashTestJob : IJob
    {
        public enum TestType
        {
            DeterminismVectorMatrix,
            SeedSensitivity,
            AvalancheElement,
            AvalancheNegation,
            LengthSensitivity,
            EmptyDeterministicSeedDependent,
            RowHashesConsistency,
            ColHashesConsistency,
        }

        public TestType Type;

        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            try
            {
                switch (Type)
                {
                    case TestType.DeterminismVectorMatrix: DeterminismVectorMatrix(ref arena); break;
                    case TestType.SeedSensitivity: SeedSensitivity(ref arena); break;
                    case TestType.AvalancheElement: AvalancheElement(ref arena); break;
                    case TestType.AvalancheNegation: AvalancheNegation(ref arena); break;
                    case TestType.LengthSensitivity: LengthSensitivity(ref arena); break;
                    case TestType.EmptyDeterministicSeedDependent: EmptyDeterministicSeedDependent(ref arena); break;
                    case TestType.RowHashesConsistency: RowHashesConsistency(ref arena); break;
                    case TestType.ColHashesConsistency: ColHashesConsistency(ref arena); break;
                    default: throw new NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        // Fills a small, deterministic, fractional test vector (fractions survive the float/double
        // expansion; they would truncate under an integer expansion, which is why the integer
        // template uses its own whole-number fills instead).
        doubleN MakeVec(ref Arena arena, int n)
        {
            var v = arena.doubleVec(n);
            for (int i = 0; i < n; i++)
                v[i] = (double)((i - 2) * 0.5f + 1.25f);
            return v;
        }

        doubleMxN MakeMat(ref Arena arena, int m, int n)
        {
            var A = arena.doubleMat(m, n);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = (double)((r * 3 - c * 2) * 0.25f + 0.5f);
            return A;
        }

        // Property 1: same input + same seed hashed twice -> identical (vector AND matrix).
        void DeterminismVectorMatrix(ref Arena arena)
        {
            var v = MakeVec(ref arena, 7);
            Assert.IsTrue(Hash.hash(in v, 0u) == Hash.hash(in v, 0u));
            Assert.IsTrue(Hash.hash(in v, 12345u) == Hash.hash(in v, 12345u));

            var A = MakeMat(ref arena, 4, 5);
            Assert.IsTrue(Hash.hash(in A, 0u) == Hash.hash(in A, 0u));
            Assert.IsTrue(Hash.hash(in A, 777u) == Hash.hash(in A, 777u));
        }

        // Property 2: same input, different seed -> different hash (for ordinary seeds).
        void SeedSensitivity(ref Arena arena)
        {
            var v = MakeVec(ref arena, 7);
            Assert.IsTrue(Hash.hash(in v, 0u) != Hash.hash(in v, 1u));
            Assert.IsTrue(Hash.hash(in v, 1u) != Hash.hash(in v, 2u));
            Assert.IsTrue(Hash.hash(in v, 0u) != Hash.hash(in v, 987654321u));

            var A = MakeMat(ref arena, 3, 4);
            Assert.IsTrue(Hash.hash(in A, 0u) != Hash.hash(in A, 42u));
        }

        // Property 3a: changing a single element changes the hash.
        void AvalancheElement(ref Arena arena)
        {
            var v = MakeVec(ref arena, 6);
            uint before = Hash.hash(in v, 0u);
            v[3] = v[3] + (double)1f; // perturb one element
            uint after = Hash.hash(in v, 0u);
            Assert.IsTrue(before != after);
        }

        // Property 3b: flipping the sign bit of one element (x -> -x, a non-zero value) changes the
        // hash. This is the numeric-value-changing form of avalanche; the pure bit-pattern forms
        // (-0.0 vs +0.0, distinct NaN payloads), which have NO numeric-value difference and are the
        // real float caveats, are pinned in the managed HashSourceTests.cs (safe from Burst folding).
        void AvalancheNegation(ref Arena arena)
        {
            var v = arena.doubleVec(4);
            v[0] = (double)1.5f; v[1] = (double)2.5f; v[2] = (double)(-3.5f); v[3] = (double)4.5f;
            uint before = Hash.hash(in v, 0u);
            v[1] = -v[1]; // 2.5 -> -2.5, flips the sign bit only
            uint after = Hash.hash(in v, 0u);
            Assert.IsTrue(before != after);
        }

        // Property 4: {1,2} and {1,2,0} (same seed) must hash differently -- the trailing zero is
        // real input, not "absence of input".
        void LengthSensitivity(ref Arena arena)
        {
            var v2 = arena.doubleVec(2);
            v2[0] = (double)1f; v2[1] = (double)2f;

            var v3 = arena.doubleVec(3);
            v3[0] = (double)1f; v3[1] = (double)2f; v3[2] = (double)0f;

            Assert.IsTrue(Hash.hash(in v2, 0u) != Hash.hash(in v3, 0u));
        }

        // Property 10: hashing a zero-length vector is deterministic and seed-DEPENDENT (not a
        // degenerate constant like always-0). Exact pinned constants are in HashSourceTests.cs.
        void EmptyDeterministicSeedDependent(ref Arena arena)
        {
            var e = arena.doubleVec(0);
            Assert.IsTrue(Hash.hash(in e, 0u) == Hash.hash(in e, 0u));      // deterministic
            Assert.IsTrue(Hash.hash(in e, 0u) != Hash.hash(in e, 12345u));  // seed-dependent
        }

        // Property 5: rowHashes[r] == hash(row r extracted as a standalone vector) for the SAME seed.
        // Exercises BOTH the allocating wrapper and the ref-dest primitive: the ref-dest call is fed
        // a buffer pre-filled with WRONG-seed hashes so a primitive that failed to overwrite some
        // row would be caught (the stale wrong-seed value would survive and mismatch).
        void RowHashesConsistency(ref Arena arena)
        {
            const uint seed = 20240704u;
            var A = MakeMat(ref arena, 5, 4);

            var alloc = Hash.rowHashes(in A, seed);
            Assert.IsTrue(alloc.N == A.M_Rows);

            // Pre-poison with a different seed's hashes, then overwrite via the ref-dest overload.
            var refDest = Hash.rowHashes(in A, seed + 999u);
            Hash.rowHashes(in A, ref refDest, seed);

            var row = arena.doubleVec(A.N_Cols);
            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    row[c] = A[r, c];
                uint expected = Hash.hash(in row, seed);
                Assert.IsTrue((int)alloc[r] == (int)expected);
                Assert.IsTrue((int)refDest[r] == (int)expected);
            }
        }

        // Property 6: colHashes[c] == hash(column c extracted as a standalone contiguous vector) for
        // the SAME seed (the strided gather must reproduce a real contiguous-vector hash).
        void ColHashesConsistency(ref Arena arena)
        {
            const uint seed = 13579u;
            var A = MakeMat(ref arena, 6, 3);

            var alloc = Hash.colHashes(in A, seed);
            Assert.IsTrue(alloc.N == A.N_Cols);

            var refDest = Hash.colHashes(in A, seed + 999u);
            Hash.colHashes(in A, ref refDest, seed);

            var col = arena.doubleVec(A.M_Rows);
            for (int c = 0; c < A.N_Cols; c++)
            {
                for (int r = 0; r < A.M_Rows; r++)
                    col[r] = A[r, c];
                uint expected = Hash.hash(in col, seed);
                Assert.IsTrue((int)alloc[c] == (int)expected);
                Assert.IsTrue((int)refDest[c] == (int)expected);
            }
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(HashTestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void HashCases(HashTestJob.TestType type)
    {
        new HashTestJob() { Type = type }.Run();
    }

    // ---- Managed contract tests: Assert.Throws must run on the main thread, not in a Burst job. ----

    // rowHashes / colHashes reject a dest whose length does not match A's row / column count. A 3x4
    // matrix gives mismatched sizes to cross-feed: rowHashes wants dest.N==3, colHashes gives N==4.
    [Test]
    public void RowColHashesWrongDestThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.doubleMat(3, 4);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    A[r, c] = (double)(r + c);

            var sizeRows = Hash.rowHashes(in A); // N == 3
            var sizeCols = Hash.colHashes(in A); // N == 4

            // rowHashes needs N==3; feeding the N==4 buffer must throw.
            Assert.Throws<ArgumentException>(() => Hash.rowHashes(in A, ref sizeCols, 0u));
            // colHashes needs N==4; feeding the N==3 buffer must throw.
            Assert.Throws<ArgumentException>(() => Hash.colHashes(in A, ref sizeRows, 0u));
        }
        finally
        {
            arena.Dispose();
        }
    }

    // A matrix with zero columns: colHashes returns an empty (N==0) buffer and touches nothing.
    [Test]
    public void ColHashesZeroColumnsReturnsEmpty()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.doubleMat(4, 0); // 4 rows, 0 cols
            var d = Hash.colHashes(in A, 0u);
            Assert.AreEqual(0, d.N);
        }
        finally
        {
            arena.Dispose();
        }
    }

    // Symmetric with ColHashesZeroColumnsReturnsEmpty: a matrix with zero rows. rowHashes has no
    // early-return guard for this case (unlike colHashes, which skips allocating its scratch gather
    // buffer) because its row loop simply iterates zero times on its own - this pins that it doesn't
    // crash and still returns an empty (N==0) buffer.
    [Test]
    public void RowHashesZeroRowsReturnsEmpty()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.doubleMat(0, 4); // 0 rows, 4 cols
            var d = Hash.rowHashes(in A, 0u);
            Assert.AreEqual(0, d.N);
        }
        finally
        {
            arena.Dispose();
        }
    }
}
