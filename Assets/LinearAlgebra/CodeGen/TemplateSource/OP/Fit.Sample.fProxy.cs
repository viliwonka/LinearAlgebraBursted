using Unity.Collections;
using Unity.Mathematics;

// Disambiguates against System.Random wherever a Fit file also imports System.
using Random = Unity.Mathematics.Random;

//+deleteThis
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

namespace BULA
{
    // ================================================================================================
    // The inverse of fitting: draw points from a shape's own surface.
    //
    // Only shapes of finite measure can do it. An infinite plane, line, cylinder or cone has no
    // uniform distribution to draw from -- the same unboundedness that stops least squares from
    // fitting their extent -- so these interfaces line up with the fitting ones rather than cutting
    // across them.
    //
    // UNIFORM means by the shape's own measure: arc length for a curve, area for a surface. The
    // obvious parameterizations are not uniform (constant-angle steps bunch at an ellipse's flat
    // sides, a lat/long grid piles up at an ellipsoid's poles), so shapes whose Jacobian varies use
    // rejection against its bound instead of the parameterization directly.
    // ================================================================================================

    /// <summary>A 3D shape that can draw points uniformly from its own surface.</summary>
    public interface IfProxySampleable3 : IfProxyShape3
    {
        /// <summary>One point, uniform by area (or by arc length, for a curve).</summary>
        fProxy3 Sample(ref Random rng);
    }

    /// <summary>A 2D shape that can draw points uniformly from its own curve.</summary>
    public interface IfProxySampleable2 : IfProxyShape2
    {
        /// <summary>One point, uniform by arc length.</summary>
        fProxy2 Sample(ref Random rng);
    }

    public static partial class Fit
    {
        /// <summary>Fills <paramref name="into"/> with uniform samples from <paramref name="shape"/>.</summary>
        public static void sample<TModel>(in TModel shape, ref Random rng, NativeArray<fProxy3> into)
            where TModel : struct, IfProxySampleable3
        {
            var s = shape;
            for (int i = 0; i < into.Length; i++) into[i] = s.Sample(ref rng);
        }

        /// <summary>Fills <paramref name="into"/> with uniform samples from <paramref name="shape"/>.</summary>
        public static void sample<TModel>(in TModel shape, ref Random rng, NativeArray<fProxy2> into)
            where TModel : struct, IfProxySampleable2
        {
            var s = shape;
            for (int i = 0; i < into.Length; i++) into[i] = s.Sample(ref rng);
        }

        // Uniform on the unit sphere. Archimedes' theorem makes this exact with no rejection: the
        // cylindrical projection of a sphere preserves area, so a uniform z and a uniform azimuth
        // already give a uniform direction.
        //
        // Reports through an OUT parameter rather than a return value on purpose. `Fit` carries no
        // per-type token, so the float and double files merge into one class; with the direction
        // returned, the two would differ only in return type, which is not an overload.
        internal static void UniformDirection(ref Random rng, out fProxy3 d)
        {
            fProxy z = rng.NextFProxy((fProxy)(-1), (fProxy)1);
            fProxy a = rng.NextFProxy((fProxy)0, (fProxy)(2.0 * math.PI_DBL));
            fProxy r = math.sqrt(math.max((fProxy)1 - z * z, (fProxy)0));
            d = new fProxy3(r * math.cos(a), r * math.sin(a), z);
        }

        // Angle whose (cos, sin) is uniform by ARC LENGTH on the ellipse (a, b). The arc element is
        // sqrt(a² sin²t + b² cos²t), which swings across the whole aspect ratio -- it is SMALLEST at
        // the major axis's vertices and largest at the minor's, so stepping t uniformly oversamples
        // the pointed ends. Rejection against max(a, b) fixes that.
        internal static fProxy EllipseAngle(ref Random rng, fProxy a, fProxy b)
        {
            fProxy big = math.max(a, b);
            fProxy t = (fProxy)0;

            for (int i = 0; i < SampleTries; i++)
            {
                t = rng.NextFProxy((fProxy)0, (fProxy)(2.0 * math.PI_DBL));
                fProxy s = math.sin(t), c = math.cos(t);
                if (rng.NextFProxy() * big <= math.sqrt(a * a * s * s + b * b * c * c)) break;
            }

            return t;
        }
    }
}
