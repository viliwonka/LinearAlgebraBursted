//singularFile//
namespace LinearAlgebra
{
    // DSP window functions for tapering a sampled signal (pre-FFT, smoothing).
    // Shared across float/double — this is a precision-independent enum, so it lives in a
    // singular file (copied verbatim) rather than being generated per-proxy.
    public enum WindowType
    {
        Box,        // rectangular: w = 1
        Hann,       // 0.5 (1 - cos(2πi/(N-1)))
        Hamming,    // 0.54 - 0.46 cos(2πi/(N-1))
        Blackman    // 0.42 - 0.5 cos(2πi/(N-1)) + 0.08 cos(4πi/(N-1))
    }
}
