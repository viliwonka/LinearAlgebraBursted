using System.IO;

using BULA;
using BULA.ML;
using BULA.Sparse;

using NUnit.Framework;
using Unity.Collections;

// Content-correctness tests for the templated debug/print surface: the fProxyPCAModel summary
// (ML/PCA.Model.fProxy.cs), the sparse block-structure printers Print.Spy / Print.Log(in fProxyBSR)
// (Sparse/Debug.Sparse.fProxy.cs), and the sparse managed exporters Print.ToText / ToCsv / SaveCsv
// (Sparse/Export.Sparse.fProxy.cs). Generated per float/double.
//
// Two surfaces get two verification styles:
//   * ToString / ToText / ToCsv return a managed string -> EXACT-string assertions (the point is
//     that the strings are RIGHT). Note the literal "fProxyPCAModel" in the model assertion is
//     itself codegen-substituted to "floatPCAModel"/"doublePCAModel", exactly as the struct's own
//     ToFixedString literal is -- so both sides move together and the match holds per type.
//   * Print.Spy / Print.Log(in fProxyBSR) are Burst-void log-only -> DoesNotThrow smoke coverage
//     (same pattern as DebugExportTests.IntLogDoesNotThrow / FloatHistogramDoesNotThrow).
//
// All of it runs on the managed test thread (ToText/ToCsv use System.Text/System.IO and cannot be
// Burst; the void logs are managed-callable too), so these are plain [Test] methods, not IJobs.
// Class is fProxy-prefixed so the generated float/double variants get distinct class names.
public class fProxyDebugPrintTests
{
    // ---------------- fProxyPCAModel summary ----------------

    // A freshly-allocated model is unconverged; the summary reports k, p (== components.M_Rows) and
    // converged=false, and NEVER dumps the component matrix.
    [Test]
    public void PcaModelUnconverged_ToStringIsExact()
    {
        var model = new fProxyPCAModel(8, 3, Allocator.Temp);   // p = 8 features, k = 3 components
        Assert.AreEqual("fProxyPCAModel(k=3, p=8, converged=false)", model.ToString());
    }

    [Test]
    public void PcaModelConvergedFlagFlipsInSummary()
    {
        var model = new fProxyPCAModel(5, 2, Allocator.Temp);
        model.converged = true;
        Assert.AreEqual("fProxyPCAModel(k=2, p=5, converged=true)", model.ToString());
    }

    // ---------------- helpers: small BSRs assembled on the managed thread ----------------

    // 2x2 block grid of 1x1 blocks (2x2 dense), block (1,0) intentionally absent:
    //   [1 2]
    //   [0 4]
    static fProxyBSR BuildNonSymmetric()
    {
        var b = new fProxyBSRBuilder(2, 2, 1, 1, Allocator.Temp);
        b.AddValue(0, 0, (fProxy)1);
        b.AddValue(0, 1, (fProxy)2);
        b.AddValue(1, 1, (fProxy)4);
        return b.ToBSR(Allocator.Temp);
    }

    // Symmetric lower-block-triangle 2x2 grid of 1x1 blocks. Stored blocks: (0,0)=5, (1,0)=3,
    // (1,1)=7; the mirror block (0,1) is NOT stored. Dense form is [[5 3][3 7]].
    static fProxyBSR BuildSymmetric()
    {
        var b = new fProxyBSRBuilder(2, 2, 1, 1, Allocator.Temp);
        b.AddValue(0, 0, (fProxy)5);
        b.AddValue(1, 0, (fProxy)3);
        b.AddValue(1, 1, (fProxy)7);
        return b.ToBSRSymmetric(Allocator.Temp);
    }

    // ---------------- sparse ToCsv (block triplet list) ----------------

    // Header "blockRow,blockCol,v0,..." plus one row per STORED block (ascending ColInd within a
    // block-row). BR*BC == 1 here -> a single value column v0.
    [Test]
    public void SparseToCsvIsBlockTripletList()
    {
        var A = BuildNonSymmetric();
        Assert.AreEqual("blockRow,blockCol,v0\n0,0,1\n0,1,2\n1,1,4\n", Print.ToCsv(in A));
    }

    // Symmetric storage's triplet CSV shows ONLY the stored lower blocks -- it does NOT mirror the
    // (0,1) block back in (unlike ToText/ToDense, which do).
    [Test]
    public void SparseToCsvSymmetricShowsOnlyStoredLowerBlocks()
    {
        var S = BuildSymmetric();
        string csv = Print.ToCsv(in S);

        Assert.AreEqual("blockRow,blockCol,v0\n0,0,5\n1,0,3\n1,1,7\n", csv);
        Assert.IsFalse(csv.Contains("\n0,1,"));   // the mirrored upper block is never emitted
    }

    // ---------------- sparse ToText (dense-ish preview) ----------------

    // ToText densifies (ToDense) then reuses the dense preview -> shows the FULL matrix including
    // the absent (1,0) block as a real zero.
    [Test]
    public void SparseToTextIsDensePreview()
    {
        var A = BuildNonSymmetric();
        Assert.AreEqual("1 2\n0 4\n", Print.ToText(in A));
    }

    // ToText DOES mirror the symmetric storage (via ToDense) -- contrast with ToCsv above, which
    // does not. Dense form of the symmetric BSR is [[5 3][3 7]].
    [Test]
    public void SparseToTextSymmetricMirrorsIntoDensePreview()
    {
        var S = BuildSymmetric();
        Assert.AreEqual("5 3\n3 7\n", Print.ToText(in S));
    }

    // ---------------- sparse SaveCsv round-trip ----------------

    [Test]
    public void SparseSaveCsvRoundTrips()
    {
        var A = BuildNonSymmetric();
        string path = Path.GetTempFileName();
        try
        {
            Print.SaveCsv(in A, path);
            Assert.AreEqual(Print.ToCsv(in A), File.ReadAllText(path));
        }
        finally { File.Delete(path); }
    }

    // ---------------- sparse Print.Spy / Print.Log smoke (Burst-void log-only) ----------------

    [Test]
    public void SparseSpyAndLogNonSymmetricDoNotThrow()
    {
        var A = BuildNonSymmetric();
        Assert.DoesNotThrow(() => Print.Spy(in A));
        Assert.DoesNotThrow(() => Print.Log(in A));
    }

    [Test]
    public void SparseSpyAndLogSymmetricDoNotThrow()
    {
        var S = BuildSymmetric();
        Assert.DoesNotThrow(() => Print.Spy(in S));   // exercises the lower->upper mirror display path
        Assert.DoesNotThrow(() => Print.Log(in S));
    }

    // Empty BSR (Nnzb == 0): every block-row's RowPtr range is empty, so the grid is all '.' and
    // the stored-block value loop never runs. Must not dereference the zero-length buffers.
    [Test]
    public void SparseSpyAndLogEmptyDoNotThrow()
    {
        var builder = new fProxyBSRBuilder(3, 3, 2, 2, Allocator.Temp);   // 6x6 dense, zero triplets
        var E = builder.ToBSR(Allocator.Temp);
        Assert.IsTrue(E.Nnzb == 0);

        Assert.DoesNotThrow(() => Print.Spy(in E));
        Assert.DoesNotThrow(() => Print.Log(in E));
    }
}
