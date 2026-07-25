using System.IO;

using BULA;

using NUnit.Framework;
using Unity.Collections;

// Content-correctness tests for the managed integer exporters (Debug/Export.iProxy.cs), generated
// per integer type (int / short / long). Integers have no round-trip precision concern, so ToText
// and ToCsv both emit the exact decimal digits (ToText space-separates matrix columns, ToCsv
// comma-separates). These helpers return System.String, so every assertion is an exact string
// match on the managed thread -- mirroring DebugExportTests' float/double cases for the new int
// coverage. Class is iProxy-prefixed so the generated int/short/long variants get distinct names.
public class iProxyDebugExportTests
{
    [Test]
    public void IntToCsvMatrixIsRowPerLineCommaSeparated()
    {
        var m = new iProxyMxN(2, 2, Allocator.Temp);
        m[0, 0] = (iProxy)1; m[0, 1] = (iProxy)2;
        m[1, 0] = (iProxy)3; m[1, 1] = (iProxy)4;

        Assert.AreEqual("1,2\n3,4\n", Print.ToCsv(in m));
    }

    [Test]
    public void IntToTextMatrixIsRowPerLineSpaceSeparated()
    {
        var m = new iProxyMxN(2, 2, Allocator.Temp);
        m[0, 0] = (iProxy)1; m[0, 1] = (iProxy)2;
        m[1, 0] = (iProxy)3; m[1, 1] = (iProxy)4;

        Assert.AreEqual("1 2\n3 4\n", Print.ToText(in m));
    }

    [Test]
    public void IntToCsvVectorIsOneValuePerLineIncludingNegatives()
    {
        var v = new iProxyN(4, Allocator.Temp);
        v[0] = (iProxy)(-2); v[1] = (iProxy)0; v[2] = (iProxy)7; v[3] = (iProxy)13;

        Assert.AreEqual("-2\n0\n7\n13\n", Print.ToCsv(in v));
    }

    [Test]
    public void IntToTextVectorHasNoTrailingNewline()
    {
        var v = new iProxyN(3, Allocator.Temp);
        v[0] = (iProxy)1; v[1] = (iProxy)2; v[2] = (iProxy)3;

        // vector ToText joins with '\n' between entries and does NOT add a trailing newline.
        Assert.AreEqual("1\n2\n3", Print.ToText(in v));
    }

    [Test]
    public void IntSaveCsvRoundTrips()
    {
        var m = new iProxyMxN(2, 3, Allocator.Temp);
        m[0, 0] = (iProxy)1; m[0, 1] = (iProxy)2; m[0, 2] = (iProxy)3;
        m[1, 0] = (iProxy)4; m[1, 1] = (iProxy)5; m[1, 2] = (iProxy)6;

        string path = Path.GetTempFileName();
        try
        {
            Print.SaveCsv(in m, path);
            Assert.AreEqual(Print.ToCsv(in m), File.ReadAllText(path));
        }
        finally { File.Delete(path); }
    }

    // int Print.Log smoke (Burst-void log-only) -- matches DebugExportTests.IntLogDoesNotThrow.
    [Test]
    public void IntLogDoesNotThrow()
    {
        var v = new iProxyN(3, Allocator.Temp);
        v[0] = (iProxy)(-2); v[1] = (iProxy)0; v[2] = (iProxy)7;
        Assert.DoesNotThrow(() => Print.Log(in v));

        var m = new iProxyMxN(2, 2, Allocator.Temp);
        m[0, 0] = (iProxy)1; m[0, 1] = (iProxy)2;
        m[1, 0] = (iProxy)3; m[1, 1] = (iProxy)4;
        Assert.DoesNotThrow(() => Print.Log(in m));
    }
}
