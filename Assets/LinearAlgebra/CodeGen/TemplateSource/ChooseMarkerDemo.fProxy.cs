namespace LinearAlgebra
{
    // Exercises the //+choose[...] codegen marker: DemoThreshold resolves to a different literal
    // per generated type. Internal: exists only for ChooseMarkerTests, not part of the library API.
    internal static class fProxyChooseMarkerDemo
    {
        public static readonly fProxy DemoThreshold = /*+choose[1e-6f|1e-14]*/1e-6f/*-choose*/;
    }
}
