using System;

using BULA;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

//alsoExpand[uint]// widens this file's int/short/long rotation to include uint, mirroring the
//production Hash.iProxy.cs so the uint vector/matrix hash + rowHashes/colHashes surface is tested
//by the same property template as int/short/long (uint rowHashes/colHashes only exist in the
//generated Hash.uint.cs, so uint is the ONLY int-family type whose bool-less overloads this reaches).

// Property tests for the integer Hash surface (int / short / long / uint) over iProxyN / iProxyMxN.
// Pure PROPERTY checks (determinism, seed sensitivity, avalanche, length sensitivity, row/col
// consistency); exact pinned hash constants live in the managed HashSourceTests.cs. All element
// fills are small NON-NEGATIVE whole numbers so the SAME template is valid for unsigned uint (no
// negative literal) and for narrow short (no overflow) as well as for int/long.
//
// CODEGEN NOTE: rowHashes/colHashes results are uint buffers (uintN) regardless of A's element type
// -- held only in `var`; elements and expected value are both funneled through `(int)x` before
// comparison (a single-step cast legal in both the raw pass and the generated code -- see the
// float/double sibling template's note). Never spell `uintN` literally: it does not exist raw.
public class iProxyHashTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct HashTestJob : IJob
    {
        public enum TestType
        {
            DeterminismVectorMatrix,
            SeedSensitivity,
            AvalancheElement,
            LengthSensitivity,
            EmptyDeterministicSeedDependent,
            RowHashesConsistency,
            ColHashesConsistency,
        }

        public TestType Type;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.DeterminismVectorMatrix: DeterminismVectorMatrix(); break;
                case TestType.SeedSensitivity: SeedSensitivity(); break;
                case TestType.AvalancheElement: AvalancheElement(); break;
                case TestType.LengthSensitivity: LengthSensitivity(); break;
                case TestType.EmptyDeterministicSeedDependent: EmptyDeterministicSeedDependent(); break;
                case TestType.RowHashesConsistency: RowHashesConsistency(); break;
                case TestType.ColHashesConsistency: ColHashesConsistency(); break;
                default: throw new NotImplementedException();
            }
        }

        // Small non-negative whole-number fills (uint-safe, short-safe).
        iProxyN MakeVec(int n)
        {
            var v = new iProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                v[i] = (iProxy)((i * 3 + 1) % 50);
            return v;
        }

        iProxyMxN MakeMat(int m, int n)
        {
            var A = new iProxyMxN(m, n, Allocator.Temp);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    A[r, c] = (iProxy)((r * 7 + c * 2 + 1) % 50);
            return A;
        }

        void DeterminismVectorMatrix()
        {
            var v = MakeVec(7);
            Assert.IsTrue(Hash.hash(in v, 0u) == Hash.hash(in v, 0u));
            Assert.IsTrue(Hash.hash(in v, 12345u) == Hash.hash(in v, 12345u));

            var A = MakeMat(4, 5);
            Assert.IsTrue(Hash.hash(in A, 0u) == Hash.hash(in A, 0u));
            Assert.IsTrue(Hash.hash(in A, 777u) == Hash.hash(in A, 777u));
        }

        void SeedSensitivity()
        {
            var v = MakeVec(7);
            Assert.IsTrue(Hash.hash(in v, 0u) != Hash.hash(in v, 1u));
            Assert.IsTrue(Hash.hash(in v, 1u) != Hash.hash(in v, 2u));
            Assert.IsTrue(Hash.hash(in v, 0u) != Hash.hash(in v, 987654321u));

            var A = MakeMat(3, 4);
            Assert.IsTrue(Hash.hash(in A, 0u) != Hash.hash(in A, 42u));
        }

        // Changing a single element changes the hash.
        void AvalancheElement()
        {
            var v = MakeVec(6);
            uint before = Hash.hash(in v, 0u);
            v[3] = (iProxy)(v[3] + (iProxy)1); // perturb one element (cast: short+short widens to int)
            uint after = Hash.hash(in v, 0u);
            Assert.IsTrue(before != after);
        }

        // {1,2} vs {1,2,0}: the trailing zero is real input.
        void LengthSensitivity()
        {
            var v2 = new iProxyN(2, Allocator.Temp);
            v2[0] = (iProxy)1; v2[1] = (iProxy)2;

            var v3 = new iProxyN(3, Allocator.Temp);
            v3[0] = (iProxy)1; v3[1] = (iProxy)2; v3[2] = (iProxy)0;

            Assert.IsTrue(Hash.hash(in v2, 0u) != Hash.hash(in v3, 0u));
        }

        void EmptyDeterministicSeedDependent()
        {
            var e = new iProxyN(0, Allocator.Temp);
            Assert.IsTrue(Hash.hash(in e, 0u) == Hash.hash(in e, 0u));
            Assert.IsTrue(Hash.hash(in e, 0u) != Hash.hash(in e, 12345u));
        }

        // rowHashes[r] == hash(row r as a standalone vector), SAME seed. Tests the allocating wrapper
        // and the ref-dest primitive (fed a wrong-seed-poisoned buffer it must fully overwrite).
        // Dimensions are deliberately ODD (5x3): for `short` (2 bytes/elem) a row is 3*2=6 bytes,
        // which is NOT a multiple of 4 - this exercises hashBytes' trailing 1-byte-at-a-time tail
        // loop (2 leftover bytes after one 4-byte group) for the short expansion of this shared
        // template. int/long/uint don't need it here (they're covered by other tests), but 3 columns
        // is a valid row width for all of them too, so the same dimensions serve every type.
        void RowHashesConsistency()
        {
            const uint seed = 20240704u;
            var A = MakeMat(5, 3);

            var alloc = Hash.rowHashes(in A, seed);
            Assert.IsTrue(alloc.N == A.M_Rows);

            var refDest = Hash.rowHashes(in A, seed + 999u);
            Hash.rowHashes(in A, ref refDest, seed);

            var row = new iProxyN(A.N_Cols, Allocator.Temp);
            for (int r = 0; r < A.M_Rows; r++)
            {
                for (int c = 0; c < A.N_Cols; c++)
                    row[c] = A[r, c];
                uint expected = Hash.hash(in row, seed);
                Assert.IsTrue((int)alloc[r] == (int)expected);
                Assert.IsTrue((int)refDest[r] == (int)expected);
            }
        }

        // colHashes[c] == hash(column c gathered into a standalone contiguous vector), SAME seed.
        void ColHashesConsistency()
        {
            const uint seed = 13579u;
            var A = MakeMat(6, 3);

            var alloc = Hash.colHashes(in A, seed);
            Assert.IsTrue(alloc.N == A.N_Cols);

            var refDest = Hash.colHashes(in A, seed + 999u);
            Hash.colHashes(in A, ref refDest, seed);

            var col = new iProxyN(A.M_Rows, Allocator.Temp);
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
        var A = new iProxyMxN(3, 4, Allocator.Temp);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                A[r, c] = (iProxy)(r + c);

        var sizeRows = Hash.rowHashes(in A); // N == 3
        var sizeCols = Hash.colHashes(in A); // N == 4

        Assert.Throws<ArgumentException>(() => Hash.rowHashes(in A, ref sizeCols, 0u));
        Assert.Throws<ArgumentException>(() => Hash.colHashes(in A, ref sizeRows, 0u));
    }

    [Test]
    public void ColHashesZeroColumnsReturnsEmpty()
    {
        var A = new iProxyMxN(4, 0, Allocator.Temp);
        var d = Hash.colHashes(in A, 0u);
        Assert.AreEqual(0, d.N);
    }

    // Symmetric with ColHashesZeroColumnsReturnsEmpty: a matrix with zero rows. rowHashes has no
    // early-return guard for this case (its row loop simply iterates zero times on its own) - pins
    // that it doesn't crash and still returns an empty (N==0) buffer.
    [Test]
    public void RowHashesZeroRowsReturnsEmpty()
    {
        var A = new iProxyMxN(0, 4, Allocator.Temp);
        var d = Hash.rowHashes(in A, 0u);
        Assert.AreEqual(0, d.N);
    }
}
