using LinearAlgebra;
using NUnit.Framework;

// Verifies the //+choose[...] codegen marker (see GenUtils.cs) resolves to the correct per-type
// literal on the short side: v0 for int, v1 for short, v2 for long - exercising the trickier
// literal-suffix rules (short needs an explicit cast, long needs the L suffix).
public class shortChooseMarkerTests
{
    [Test]
    public void ChooseMarker_ResolvesToCorrectPerTypeLiteral()
    {
        // shortChooseMarkerDemo.DemoValue is declared in the template as
        //   (short)100
        // which must resolve to 100 in this file regardless of which short type (int/short/long).
        Assert.AreEqual((short)100, shortChooseMarkerDemo.DemoValue);
    }
}
