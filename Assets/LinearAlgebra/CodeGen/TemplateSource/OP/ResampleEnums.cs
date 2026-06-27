//singularFile//
namespace LinearAlgebra
{
    // Interpolation mode for ResampleOP.
    // Nearest = round pos to nearest integer index (no cross-sample blending).
    // Linear  = lerp between the two bracketing samples (C0 continuous).
    // Cubic   = Catmull-Rom 4-point stencil (interpolating, C1 continuous, passes through data).
    public enum Interp { Nearest, Linear, Cubic }

    // Edge mode for ResampleOP — how out-of-range tap indices are resolved.
    // Clamp  = repeat the nearest edge sample:  clamp(i, 0, N-1).
    // Wrap   = periodic / tiling:               ((i % N) + N) % N.
    // Mirror = no-edge-repeat reflection (reflect101 / OpenCV border) with period 2*(N-1);
    //          for N==1 always returns index 0.
    public enum EdgeMode { Clamp, Wrap, Mirror }
}
