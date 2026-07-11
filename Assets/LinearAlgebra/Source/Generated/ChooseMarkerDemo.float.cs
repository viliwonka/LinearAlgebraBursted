namespace LinearAlgebra
{
    // Exercises the //+choose[...] codegen marker: DemoThreshold resolves to a different literal
    // per generated type. Internal: exists only for ChooseMarkerTests, not part of the library API.
    internal static class floatChooseMarkerDemo
    {
        public static readonly float DemoThreshold = 1e-6f;
    }
}
