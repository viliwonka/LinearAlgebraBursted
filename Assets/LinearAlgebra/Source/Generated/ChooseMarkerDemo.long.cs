namespace LinearAlgebra
{
    // Demonstrates/exercises the //+choose[...] codegen marker (see GenUtils.cs and
    // TemplateConverter.ChooseReplace) on the long (int/short/long) side, where literal suffix
    // rules differ per type (int needs none, short needs an explicit cast, long needs the L suffix).
    // Covered by ChooseMarkerTests.long.cs. Not part of the library's public surface.
    public static class longChooseMarkerDemo
    {
        public static readonly long DemoValue = 100L;
    }
}
