using System.IO;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Collections;

// Concrete (NOT codegen'd) tests for the managed Print export helpers and the new int / histogram
// dumps. They live here rather than in TemplateSourceTests because they exercise the concrete
// float / double / int types and the hand-authored Print.ToCsv / ToText overloads (which cannot be
// templated — the proxy type has no formatted ToString). The CSV / ToText helpers return strings so
// they are directly assertable; the Burst Log / Histogram dumps only log, so they get smoke coverage.
public class DebugExportTests
{
    // ---------------- float ----------------

    [Test]
    public void FloatToCsvVectorIsOneValuePerLine()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.floatVec(3);
        v[0] = 1; v[1] = 2; v[2] = 3;

        Assert.AreEqual("1\n2\n3\n", Print.ToCsv(in v));

        arena.Dispose();
    }

    [Test]
    public void FloatToCsvMatrixIsRowPerLineCommaSeparated()
    {
        var arena = new Arena(Allocator.Persistent);
        var m = arena.floatMat(2, 2);
        m[0, 0] = 1; m[0, 1] = 2;
        m[1, 0] = 3; m[1, 1] = 4;

        Assert.AreEqual("1,2\n3,4\n", Print.ToCsv(in m));

        arena.Dispose();
    }

    [Test]
    public void FloatToCsvUsesInvariantDecimalPoint()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.floatVec(1);
        v[0] = 1.5f;

        Assert.AreEqual("1.5\n", Print.ToCsv(in v));

        arena.Dispose();
    }

    [Test]
    public void FloatToTextDoesNotTruncateLargeMatrix()
    {
        var arena = new Arena(Allocator.Persistent);
        var m = arena.floatMat(100, 100);
        for (int r = 0; r < 100; r++)
            for (int c = 0; c < 100; c++)
                m[r, c] = r * 100 + c;

        string text = Print.ToText(in m);

        Assert.Greater(text.Length, 4096);                 // Burst Print.Log would have capped at 4 KB
        Assert.AreEqual(100, text.Split('\n').Length - 1); // 100 rows, trailing newline

        arena.Dispose();
    }

    [Test]
    public void FloatSaveCsvRoundTrips()
    {
        var arena = new Arena(Allocator.Persistent);
        var m = arena.floatMat(2, 3);
        m[0, 0] = 1; m[0, 1] = 2; m[0, 2] = 3;
        m[1, 0] = 4; m[1, 1] = 5; m[1, 2] = 6;

        string path = Path.GetTempFileName();
        try
        {
            Print.SaveCsv(in m, path);
            Assert.AreEqual(Print.ToCsv(in m), File.ReadAllText(path));
        }
        finally { File.Delete(path); }

        arena.Dispose();
    }

    [Test]
    public void FloatHistogramDoesNotThrow()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.floatVec(64);
        for (int i = 0; i < 64; i++) v[i] = i;
        Assert.DoesNotThrow(() => Print.Histogram(in v, 8, 20));

        var flat = arena.floatVec(10);          // all identical -> range 0 branch
        for (int i = 0; i < 10; i++) flat[i] = 5;
        Assert.DoesNotThrow(() => Print.Histogram(in flat, 8, 20));

        arena.Dispose();
    }

    // ---------------- double ----------------

    [Test]
    public void DoubleToCsvVectorIsOneValuePerLine()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.doubleVec(3);
        v[0] = 1; v[1] = 2; v[2] = 3;

        Assert.AreEqual("1\n2\n3\n", Print.ToCsv(in v));

        arena.Dispose();
    }

    [Test]
    public void DoubleToCsvMatrixIsRowPerLineCommaSeparated()
    {
        var arena = new Arena(Allocator.Persistent);
        var m = arena.doubleMat(2, 2);
        m[0, 0] = 1; m[0, 1] = 2;
        m[1, 0] = 3; m[1, 1] = 4;

        Assert.AreEqual("1,2\n3,4\n", Print.ToCsv(in m));

        arena.Dispose();
    }

    [Test]
    public void DoubleSaveCsvRoundTrips()
    {
        var arena = new Arena(Allocator.Persistent);
        var v = arena.doubleVec(4);
        v[0] = 1; v[1] = 2; v[2] = 3; v[3] = 4;

        string path = Path.GetTempFileName();
        try
        {
            Print.SaveCsv(in v, path);
            Assert.AreEqual(Print.ToCsv(in v), File.ReadAllText(path));
        }
        finally { File.Delete(path); }

        arena.Dispose();
    }

    // ---------------- int (new Log overloads) ----------------

    [Test]
    public void IntLogDoesNotThrow()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.intVec(4);
        v[0] = -2; v[1] = 0; v[2] = 7; v[3] = 13;
        Assert.DoesNotThrow(() => Print.Log(in v));

        var m = arena.intMat(2, 2);
        m[0, 0] = 1; m[0, 1] = 2;
        m[1, 0] = 3; m[1, 1] = 4;
        Assert.DoesNotThrow(() => Print.Log(in m));

        arena.Dispose();
    }
}
