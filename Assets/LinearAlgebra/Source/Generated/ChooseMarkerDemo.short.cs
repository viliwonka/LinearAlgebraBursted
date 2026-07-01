namespace LinearAlgebra
{
    // Demonstrates/exercises the //+choose[...] codegen marker (see GenUtils.cs and
    // TemplateConverter.ChooseReplace) on the short (int/short/long) side, where literal suffix
    // rules differ per type (int needs none, short needs an explicit cast, long needs the L suffix).
    // Covered by ChooseMarkerTests.short.cs. Not part of the library's public surface.
    public static class shortChooseMarkerDemo
    {
        public static readonly short DemoValue = (short)100;
    }
}
