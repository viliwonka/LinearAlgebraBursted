using System.IO;

using LinearAlgebra;
using LinearAlgebra.ML;
using LinearAlgebra.Sparse;

using NUnit.Framework;
using Unity.Collections;

// Content-correctness tests for the templated debug/print surface: the floatPCAModel summary
// (ML/PCA.Model.float.cs), the sparse block-structure printers Print.Spy / Print.Log(in floatBSR)
// (Sparse/Debug.Sparse.float.cs), and the sparse managed exporters Print.ToText / ToCsv / SaveCsv
// (Sparse/Export.Sparse.float.cs). Generated per float/double.
//
// Two surfaces get two verification styles:
//   * ToString / ToText / ToCsv return a managed string -> EXACT-string assertions (the point is
//     that the strings are RIGHT). Note the literal "floatPCAModel" in the model assertion is
//     itself codegen-substituted to "floatPCAModel"/"doublePCAModel", exactly as the struct's own
//     ToFixedString literal is -- so both sides move together and the match holds per type.
//   * Print.Spy / Print.Log(in floatBSR) are Burst-void log-only -> DoesNotThrow smoke coverage
//     (same pattern as DebugExportTests.IntLogDoesNotThrow / FloatHistogramDoesNotThrow).
//
// All of it runs on the managed test thread (ToText/ToCsv use System.Text/System.IO and cannot be
// Burst; the void logs are managed-callable too), so these are plain [Test] methods, not IJobs.
// Class is float-prefixed so the generated float/double variants get distinct class names.
public class floatDebugPrintTests
{
    // ---------------- floatPCAModel summary ----------------

    // A freshly-allocated model is unconverged; the summary reports k, p (== components.M_Rows) and
    // converged=false, and NEVER dumps the component matrix.
    [Test]
    public void PcaModelUnconverged_ToStringIsExact()
    {
        var arena = new Arena(Allocator.Persistent);

        var model = arena.floatPCAModel(8, 3);   // p = 8 features, k = 3 components
        Assert.AreEqual("floatPCAModel(k=3, p=8, converged=false)", model.ToString());

        arena.Dispose();
    }

    [Test]
    public void PcaModelConvergedFlagFlipsInSummary()
    {
        var arena = new Arena(Allocator.Persistent);

        var model = arena.floatPCAModel(5, 2);
        model.converged = true;
        Assert.AreEqual("floatPCAModel(k=2, p=5, converged=true)", model.ToString());

        arena.Dispose();
    }

    // ---------------- helpers: small BSRs assembled on the managed thread ----------------

    // 2x2 block grid of 1x1 blocks (2x2 dense), block (1,0) intentionally absent:
    //   [1 2]
    //   [0 4]
    static floatBSR BuildNonSymmetric(ref Arena arena)
    {
        var b = arena.floatBSRBuilder(2, 2, 1, 1);
        b.AddValue(0, 0, (float)1);
        b.AddValue(0, 1, (float)2);
        b.AddValue(1, 1, (float)4);
        return b.ToBSR(ref arena);
    }

    // Symmetric upper-block-triangle 2x2 grid of 1x1 blocks. Stored blocks: (0,0)=5, (0,1)=3,
    // (1,1)=7; the mirror block (1,0) is NOT stored. Dense form is [[5 3][3 7]].
    static floatBSR BuildSymmetric(ref Arena arena)
    {
        var b = arena.floatBSRBuilder(2, 2, 1, 1);
        b.AddValue(0, 0, (float)5);
        b.AddValue(0, 1, (float)3);
        b.AddValue(1, 1, (float)7);
        return b.ToBSRSymmetric(ref arena);
    }

    // ---------------- sparse ToCsv (block triplet list) ----------------

    // Header "blockRow,blockCol,v0,..." plus one row per STORED block (ascending ColInd within a
    // block-row). BR*BC == 1 here -> a single value column v0.
    [Test]
    public void SparseToCsvIsBlockTripletList()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = BuildNonSymmetric(ref arena);
        Assert.AreEqual("blockRow,blockCol,v0\n0,0,1\n0,1,2\n1,1,4\n", Print.ToCsv(in A));

        arena.Dispose();
    }

    // Symmetric storage's triplet CSV shows ONLY the stored upper blocks -- it does NOT mirror the
    // (1,0) block back in (unlike ToText/ToDense, which do).
    [Test]
    public void SparseToCsvSymmetricShowsOnlyStoredUpperBlocks()
    {
        var arena = new Arena(Allocator.Persistent);

        var S = BuildSymmetric(ref arena);
        string csv = Print.ToCsv(in S);

        Assert.AreEqual("blockRow,blockCol,v0\n0,0,5\n0,1,3\n1,1,7\n", csv);
        Assert.IsFalse(csv.Contains("\n1,0,"));   // the mirrored lower block is never emitted

        arena.Dispose();
    }

    // ---------------- sparse ToText (dense-ish preview) ----------------

    // ToText densifies (ToDense) then reuses the dense preview -> shows the FULL matrix including
    // the absent (1,0) block as a real zero.
    [Test]
    public void SparseToTextIsDensePreview()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = BuildNonSymmetric(ref arena);
        Assert.AreEqual("1 2\n0 4\n", Print.ToText(in A));

        arena.Dispose();
    }

    // ToText DOES mirror the symmetric storage (via ToDense) -- contrast with ToCsv above, which
    // does not. Dense form of the symmetric BSR is [[5 3][3 7]].
    [Test]
    public void SparseToTextSymmetricMirrorsIntoDensePreview()
    {
        var arena = new Arena(Allocator.Persistent);

        var S = BuildSymmetric(ref arena);
        Assert.AreEqual("5 3\n3 7\n", Print.ToText(in S));

        arena.Dispose();
    }

    // ---------------- sparse SaveCsv round-trip ----------------

    [Test]
    public void SparseSaveCsvRoundTrips()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = BuildNonSymmetric(ref arena);
        string path = Path.GetTempFileName();
        try
        {
            Print.SaveCsv(in A, path);
            Assert.AreEqual(Print.ToCsv(in A), File.ReadAllText(path));
        }
        finally { File.Delete(path); }

        arena.Dispose();
    }

    // ---------------- sparse Print.Spy / Print.Log smoke (Burst-void log-only) ----------------

    [Test]
    public void SparseSpyAndLogNonSymmetricDoNotThrow()
    {
        var arena = new Arena(Allocator.Persistent);

        var A = BuildNonSymmetric(ref arena);
        Assert.DoesNotThrow(() => Print.Spy(in A));
        Assert.DoesNotThrow(() => Print.Log(in A));

        arena.Dispose();
    }

    [Test]
    public void SparseSpyAndLogSymmetricDoNotThrow()
    {
        var arena = new Arena(Allocator.Persistent);

        var S = BuildSymmetric(ref arena);
        Assert.DoesNotThrow(() => Print.Spy(in S));   // exercises the upper->lower mirror display path
        Assert.DoesNotThrow(() => Print.Log(in S));

        arena.Dispose();
    }

    // Empty BSR (Nnzb == 0): every block-row's RowPtr range is empty, so the grid is all '.' and
    // the stored-block value loop never runs. Must not dereference the zero-length buffers.
    [Test]
    public void SparseSpyAndLogEmptyDoNotThrow()
    {
        var arena = new Arena(Allocator.Persistent);

        var builder = arena.floatBSRBuilder(3, 3, 2, 2);   // 6x6 dense, zero triplets
        var E = builder.ToBSR(ref arena);
        Assert.IsTrue(E.Nnzb == 0);

        Assert.DoesNotThrow(() => Print.Spy(in E));
        Assert.DoesNotThrow(() => Print.Log(in E));

        arena.Dispose();
    }
}
