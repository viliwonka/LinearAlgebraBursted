namespace LinearAlgebra
{
    // Demonstrates/exercises the //+choose[...] codegen marker (see GenUtils.cs and
    // TemplateConverter.ChooseReplace): DemoThreshold resolves to a DIFFERENT literal per generated
    // type, not just a different TYPE NAME the way plain double substitution does. Covered by
    // ChooseMarkerTests.double.cs. Not part of the library's public surface.
    public static class doubleChooseMarkerDemo
    {
        public static readonly double DemoThreshold = 1e-14;
    }
}
