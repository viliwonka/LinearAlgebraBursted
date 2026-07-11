namespace LinearAlgebra
{
    // Exercises the //+choose[...] codegen marker on the int/short/long side, where literal suffix
    // rules differ per type. Internal: exists only for ChooseMarkerTests, not part of the library API.
    internal static class longChooseMarkerDemo
    {
        public static readonly long DemoValue = 100L;
    }
}
