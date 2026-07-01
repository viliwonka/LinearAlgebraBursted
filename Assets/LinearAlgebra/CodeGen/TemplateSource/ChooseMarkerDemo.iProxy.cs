namespace LinearAlgebra
{
    // Demonstrates/exercises the //+choose[...] codegen marker (see GenUtils.cs and
    // TemplateConverter.ChooseReplace) on the iProxy (int/short/long) side, where literal suffix
    // rules differ per type (int needs none, short needs an explicit cast, long needs the L suffix).
    // Covered by ChooseMarkerTests.iProxy.cs. Not part of the library's public surface.
    public static class iProxyChooseMarkerDemo
    {
        public static readonly iProxy DemoValue = /*+choose[100|(short)100|100L]*/100/*-choose*/;
    }
}
