using Unity.Collections;

using LinearAlgebra.ML;

namespace LinearAlgebra
{
    // Burst-safe Print.Log for the PCA model buffer struct (ML/PCA.Model.float.cs). Mirrors the
    // rest of Print.Log's style (Debug/Debug.float.cs): FixedString + UnityEngine.Debug.Log.
    // Templated (float/double) because floatPCAModel itself is templated. Logs the compact
    // dims/k/converged summary only -- never the (unbounded) components matrix.
    public static partial class Print
    {
        public static void Log(in LinearAlgebra.ML.floatPCAModel model)
        {
            FixedString128Bytes str = model.ToFixedString();
            UnityEngine.Debug.Log($"{str}");
        }
    }
}
