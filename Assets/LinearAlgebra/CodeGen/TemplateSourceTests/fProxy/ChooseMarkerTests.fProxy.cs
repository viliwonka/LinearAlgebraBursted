using BULA;
using NUnit.Framework;

// Verifies the //+choose[...] codegen marker (see GenUtils.cs) resolves to the correct per-type
// literal: v0 for the first type in the file's types[] array (float), v1 for the second (double).
public class fProxyChooseMarkerTests
{
    [Test]
    public void ChooseMarker_ResolvesToCorrectPerTypeLiteral()
    {
        // fProxyChooseMarkerDemo.DemoThreshold is declared in the template as
        //   /*+choose[1e-6f|1e-14]*/1e-6f/*-choose*/
        // which must resolve to 1e-6f in this (float) file and 1e-14 in the double file - use the
        // SAME marker for the expected value so both generated tests compare against the right one
        // (a hardcoded "1e-6f" here would be wrong once cast to double: it isn't 1e-14).
        fProxy expected = /*+choose[1e-6f|1e-14]*/1e-6f/*-choose*/;
        Assert.AreEqual(expected, fProxyChooseMarkerDemo.DemoThreshold);
    }
}
