using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// A sampler draws one random value per call, advancing the caller-owned RNG stream.
    /// Implementations may carry mutable state (e.g. <see cref="floatGaussian"/> caches one
    /// spare variate per Box–Muller pair); pass by <c>ref</c> so that state changes persist
    /// across elements when used with <c>Rand.randomInPlace</c>.
    /// Implement on a blittable struct (only blittable fields, no managed references) for
    /// Burst compatibility — same contract as <see cref="IfloatScalarFunction"/>.
    /// </summary>
    public interface IfloatSampler
    {
        float Next(ref Random rng);
    }
}
