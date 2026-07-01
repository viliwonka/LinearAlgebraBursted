using LinearAlgebra;
using NUnit.Framework;

// Verifies the //+choose[...] codegen marker (see GenUtils.cs) resolves to the correct per-type
// literal on the iProxy side: v0 for int, v1 for short, v2 for long - exercising the trickier
// literal-suffix rules (short needs an explicit cast, long needs the L suffix).
public class iProxyChooseMarkerTests
{
    [Test]
    public void ChooseMarker_ResolvesToCorrectPerTypeLiteral()
    {
        // iProxyChooseMarkerDemo.DemoValue is declared in the template as
        //   /*+choose[100|(short)100|100L]*/100/*-choose*/
        // which must resolve to 100 in this file regardless of which iProxy type (int/short/long).
        Assert.AreEqual((iProxy)100, iProxyChooseMarkerDemo.DemoValue);
    }
}
