using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// Periodic wave functors (each : IfProxyScalarFunction) for building wavetables / LFOs. Eval(t)
    /// treats t∈[0,1] as <c>Cycles</c> full periods (Phase shifts the start). Output range is [-1,1].
    /// Bake a table with <c>var w = new fProxyWave.Sine{Cycles=1}; Generate.sample(ref w, ref dest);</c>.
    /// fProxy-only.
    ///
    /// Default-construction convenience: a <c>Cycles</c> of 0 is treated as 1 (and Square's <c>Duty</c>
    /// of 0 as 0.5) so that <c>new fProxyWave.Sine()</c> is a usable 1-cycle wave rather than a flat
    /// line — set <c>Cycles</c> explicitly. There is therefore no way to request literally 0 cycles.
    /// </summary>
    public static partial class fProxyWave
    {
        /// <summary>Sine wave: sin(2π(Cycles·t + Phase)). Defaults: Cycles=1, Phase=0.</summary>
        public struct Sine : IfProxyScalarFunction
        {
            public fProxy Cycles;
            public fProxy Phase;

            public fProxy Eval(fProxy t)
            {
                fProxy cyc = Cycles == (fProxy)0 ? (fProxy)1 : Cycles;
                return DetMath.Sin((fProxy)(2.0 * System.Math.PI) * (cyc * t + Phase));
            }
        }

        /// <summary>Rising sawtooth, ramping from -1 up to (but not including) +1 each period — i.e. [-1,1). At a period boundary (frac==0) the value is -1. Defaults: Cycles=1, Phase=0.</summary>
        public struct Saw : IfProxyScalarFunction
        {
            public fProxy Cycles;
            public fProxy Phase;

            public fProxy Eval(fProxy t)
            {
                fProxy cyc = Cycles == (fProxy)0 ? (fProxy)1 : Cycles;
                fProxy x = cyc * t + Phase;
                fProxy frac = x - math.floor(x);       // [0,1)
                return (fProxy)2 * frac - (fProxy)1;    // [-1,1)
            }
        }

        /// <summary>Square wave: +1 for the first <c>Duty</c> fraction of each period, else -1. Defaults: Cycles=1, Duty=0.5, Phase=0.
        /// Note: a <c>Duty</c> of exactly 0 is treated as the 0.5 default (see the class remarks), so an always-low square can't be requested this way.</summary>
        public struct Square : IfProxyScalarFunction
        {
            public fProxy Cycles;
            public fProxy Duty;
            public fProxy Phase;

            public fProxy Eval(fProxy t)
            {
                fProxy cyc = Cycles == (fProxy)0 ? (fProxy)1 : Cycles;
                fProxy duty = Duty == (fProxy)0 ? (fProxy)0.5 : Duty;
                fProxy x = cyc * t + Phase;
                fProxy frac = x - math.floor(x);        // [0,1)
                return frac < duty ? (fProxy)1 : (fProxy)(-1);
            }
        }

        /// <summary>Triangle wave in [-1,1]; -1 at period edges, peaking at +1 where frac(Cycles·t+Phase)=0.5 (the period midpoint when Phase=0). Defaults: Cycles=1, Phase=0.</summary>
        public struct Triangle : IfProxyScalarFunction
        {
            public fProxy Cycles;
            public fProxy Phase;

            public fProxy Eval(fProxy t)
            {
                fProxy cyc = Cycles == (fProxy)0 ? (fProxy)1 : Cycles;
                fProxy x = cyc * t + Phase;
                fProxy frac = x - math.floor(x);             // [0,1)
                return (fProxy)1 - (fProxy)4 * math.abs(frac - (fProxy)0.5); // -1 at edges, +1 at midpoint
            }
        }
    }
}
