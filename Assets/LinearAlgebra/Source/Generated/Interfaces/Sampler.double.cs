using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// A sampler draws one random value per call, advancing the caller-owned RNG stream.
    /// Implementations may carry mutable state (e.g. <see cref="doubleGaussian"/> caches one
    /// spare variate per Box–Muller pair); pass by <c>ref</c> so that state changes persist
    /// across elements when used with <c>Rand.randomInpl</c>.
    /// Implement on a blittable struct (only blittable fields, no managed references) for
    /// Burst compatibility — same contract as <see cref="IdoubleScalarFunction"/>.
    /// </summary>
    public interface IdoubleSampler
    {
        double Next(ref Random rng);
    }
}
