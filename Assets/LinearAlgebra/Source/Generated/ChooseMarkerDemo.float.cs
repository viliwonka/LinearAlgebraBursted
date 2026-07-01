namespace LinearAlgebra
{
    // Demonstrates/exercises the //+choose[...] codegen marker (see GenUtils.cs and
    // TemplateConverter.ChooseReplace): DemoThreshold resolves to a DIFFERENT literal per generated
    // type, not just a different TYPE NAME the way plain float substitution does. Covered by
    // ChooseMarkerTests.float.cs. Not part of the library's public surface.
    public static class floatChooseMarkerDemo
    {
        public static readonly float DemoThreshold = 1e-6f;
    }
}
