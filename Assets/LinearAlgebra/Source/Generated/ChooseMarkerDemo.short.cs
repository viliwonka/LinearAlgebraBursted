namespace LinearAlgebra
{
    // Exercises the //+choose[...] codegen marker on the int/short/long side, where literal suffix
    // rules differ per type. Internal: exists only for ChooseMarkerTests, not part of the library API.
    internal static class shortChooseMarkerDemo
    {
        public static readonly short DemoValue = (short)100;
    }
}
