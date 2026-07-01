namespace LinearAlgebra
{
    // Demonstrates/exercises the //+choose[...] codegen marker (see GenUtils.cs and
    // TemplateConverter.ChooseReplace): DemoThreshold resolves to a DIFFERENT literal per generated
    // type, not just a different TYPE NAME the way plain fProxy substitution does. Covered by
    // ChooseMarkerTests.fProxy.cs. Not part of the library's public surface.
    public static class fProxyChooseMarkerDemo
    {
        public static readonly fProxy DemoThreshold = /*+choose[1e-6f|1e-14]*/1e-6f/*-choose*/;
    }
}
