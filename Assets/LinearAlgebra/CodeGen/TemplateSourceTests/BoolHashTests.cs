using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Property tests for the bool Hash surface (Hash.hash / rowHashes / colHashes over boolN / boolMxN).
// bool is a genuinely singular type (no codegen rotation), and its rowHashes/colHashes overloads
// live in the int-family file's skipFor'd uint slot but merge into the same `Hash` class -- from a
// consumer's view they are ordinary overloads, tested here like any other type. Burst stores a bool
// as one byte (0/1), so equal-valued bools always hash identically (no -0.0/NaN caveat). Exact
// pinned constants are shared across types in the managed HashSourceTests.cs.
public class BoolHashTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct HashTestJob : IJob
    {
        public enum TestType
        {
            DeterminismVectorMatrix,
            SeedSensitivity,
            AvalancheBitFlip,
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
                    case TestType.AvalancheBitFlip: AvalancheBitFlip(ref arena); break;
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

        boolN MakeVec(ref Arena arena, int n)
        {
            var v = arena.boolVec(n);
            for (int i = 0; i < n; i++)
                v[i] = ((i * 5 + 2) % 3) == 0; // deterministic pseudo-pattern
            return v;
        }

        boolMxN MakeMat(ref Arena arena, int m, int n)
        {
            var A = arena.boolMat(m, n);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = ((r * 3 + c * 2) % 2) == 0;
            return A;
        }

        void DeterminismVectorMatrix(ref Arena arena)
        {
            var v = MakeVec(ref arena, 9);
            Assert.IsTrue(Hash.hash(in v, 0u) == Hash.hash(in v, 0u));
            Assert.IsTrue(Hash.hash(in v, 12345u) == Hash.hash(in v, 12345u));

            var A = MakeMat(ref arena, 4, 5);
            Assert.IsTrue(Hash.hash(in A, 0u) == Hash.hash(in A, 0u));
            Assert.IsTrue(Hash.hash(in A, 777u) == Hash.hash(in A, 777u));
        }

        void SeedSensitivity(ref Arena arena)
        {
            var v = MakeVec(ref arena, 9);
            Assert.IsTrue(Hash.hash(in v, 0u) != Hash.hash(in v, 1u));
            Assert.IsTrue(Hash.hash(in v, 0u) != Hash.hash(in v, 987654321u));

            var A = MakeMat(ref arena, 3, 4);
            Assert.IsTrue(Hash.hash(in A, 0u) != Hash.hash(in A, 42u));
        }

        // Flipping a single bool element changes the hash.
        void AvalancheBitFlip(ref Arena arena)
        {
            var v = MakeVec(ref arena, 8);
            uint before = Hash.hash(in v, 0u);
            v[3] = !v[3]; // flip one element
            uint after = Hash.hash(in v, 0u);
            Assert.IsTrue(before != after);
        }

        // {true,true} vs {true,true,false}: the trailing false byte is real input.
        void LengthSensitivity(ref Arena arena)
        {
            var v2 = arena.boolVec(2);
            v2[0] = true; v2[1] = true;

            var v3 = arena.boolVec(3);
            v3[0] = true; v3[1] = true; v3[2] = false;

            Assert.IsTrue(Hash.hash(in v2, 0u) != Hash.hash(in v3, 0u));
        }

        void EmptyDeterministicSeedDependent(ref Arena arena)
        {
            var e = arena.boolVec(0);
            Assert.IsTrue(Hash.hash(in e, 0u) == Hash.hash(in e, 0u));
            Assert.IsTrue(Hash.hash(in e, 0u) != Hash.hash(in e, 12345u));
        }

        void RowHashesConsistency(ref Arena arena)
        {
            const uint seed = 20240704u;
            var A = MakeMat(ref arena, 5, 4);

            var alloc = Hash.rowHashes(in A, seed);
            Assert.IsTrue(alloc.N == A.M_Rows);

            var refDest = Hash.rowHashes(in A, seed + 999u);
            Hash.rowHashes(in A, ref refDest, seed);

            var row = arena.boolVec(A.N_Cols);
            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    row[c] = A[r, c];
                uint expected = Hash.hash(in row, seed);
                Assert.IsTrue((int)alloc[r] == (int)expected);
                Assert.IsTrue((int)refDest[r] == (int)expected);
            }
        }

        void ColHashesConsistency(ref Arena arena)
        {
            const uint seed = 13579u;
            var A = MakeMat(ref arena, 6, 3);

            var alloc = Hash.colHashes(in A, seed);
            Assert.IsTrue(alloc.N == A.N_Cols);

            var refDest = Hash.colHashes(in A, seed + 999u);
            Hash.colHashes(in A, ref refDest, seed);

            var col = arena.boolVec(A.M_Rows);
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

    // ---- Managed contract tests (Assert.Throws off the Burst thread). ----

    [Test]
    public void RowColHashesWrongDestThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.boolMat(3, 4);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    A[r, c] = ((r + c) % 2) == 0;

            var sizeRows = Hash.rowHashes(in A); // N == 3
            var sizeCols = Hash.colHashes(in A); // N == 4

            Assert.Throws<ArgumentException>(() => Hash.rowHashes(in A, ref sizeCols, 0u));
            Assert.Throws<ArgumentException>(() => Hash.colHashes(in A, ref sizeRows, 0u));
        }
        finally
        {
            arena.Dispose();
        }
    }

    [Test]
    public void ColHashesZeroColumnsReturnsEmpty()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.boolMat(4, 0);
            var d = Hash.colHashes(in A, 0u);
            Assert.AreEqual(0, d.N);
        }
        finally
        {
            arena.Dispose();
        }
    }

    // Symmetric with ColHashesZeroColumnsReturnsEmpty: a matrix with zero rows. rowHashes has no
    // early-return guard for this case (its row loop simply iterates zero times on its own) - pins
    // that it doesn't crash and still returns an empty (N==0) buffer.
    [Test]
    public void RowHashesZeroRowsReturnsEmpty()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var A = arena.boolMat(0, 4);
            var d = Hash.rowHashes(in A, 0u);
            Assert.AreEqual(0, d.N);
        }
        finally
        {
            arena.Dispose();
        }
    }
}
