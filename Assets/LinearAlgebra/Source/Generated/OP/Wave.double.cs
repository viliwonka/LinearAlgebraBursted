#define UNITY_BURST_EXPERIMENTAL_LOOP_INTRINSICS

using Unity.Mathematics;

namespace LinearAlgebra
{
    /// <summary>
    /// Periodic wave functors (each : IdoubleScalarFunction) for building wavetables / LFOs. Eval(t)
    /// treats t∈[0,1] as <c>Cycles</c> full periods (Phase shifts the start). Output range is [-1,1].
    /// Bake a table with <c>var w = new doubleWave.Sine{Cycles=1}; doubleGenOP.sample(ref w, ref dest);</c>
    /// or <c>arena.doubleSample(ref w, n)</c>. double-only.
    ///
    /// Default-construction convenience: a <c>Cycles</c> of 0 is treated as 1 (and Square's <c>Duty</c>
    /// of 0 as 0.5) so that <c>new doubleWave.Sine()</c> is a usable 1-cycle wave rather than a flat
    /// line — set <c>Cycles</c> explicitly. There is therefore no way to request literally 0 cycles.
    /// </summary>
    public static partial class doubleWave
    {
        /// <summary>Sine wave: sin(2π(Cycles·t + Phase)). Defaults: Cycles=1, Phase=0.</summary>
        public struct Sine : IdoubleScalarFunction
        {
            public double Cycles;
            public double Phase;

            public double Eval(double t)
            {
                double cyc = Cycles == (double)0 ? (double)1 : Cycles;
                return math.sin((double)(2.0 * System.Math.PI) * (cyc * t + Phase));
            }
        }

        /// <summary>Rising sawtooth, ramping from -1 up to (but not including) +1 each period — i.e. [-1,1). At a period boundary (frac==0) the value is -1. Defaults: Cycles=1, Phase=0.</summary>
        public struct Saw : IdoubleScalarFunction
        {
            public double Cycles;
            public double Phase;

            public double Eval(double t)
            {
                double cyc = Cycles == (double)0 ? (double)1 : Cycles;
                double x = cyc * t + Phase;
                double frac = x - math.floor(x);       // [0,1)
                return (double)2 * frac - (double)1;    // [-1,1)
            }
        }

        /// <summary>Square wave: +1 for the first <c>Duty</c> fraction of each period, else -1. Defaults: Cycles=1, Duty=0.5, Phase=0.</summary>
        public struct Square : IdoubleScalarFunction
        {
            public double Cycles;
            public double Duty;
            public double Phase;

            public double Eval(double t)
            {
                double cyc = Cycles == (double)0 ? (double)1 : Cycles;
                double duty = Duty == (double)0 ? (double)0.5 : Duty;
                double x = cyc * t + Phase;
                double frac = x - math.floor(x);        // [0,1)
                return frac < duty ? (double)1 : (double)(-1);
            }
        }

        /// <summary>Triangle wave in [-1,1]; -1 at period edges, peaking at +1 where frac(Cycles·t+Phase)=0.5 (the period midpoint when Phase=0). Defaults: Cycles=1, Phase=0.</summary>
        public struct Triangle : IdoubleScalarFunction
        {
            public double Cycles;
            public double Phase;

            public double Eval(double t)
            {
                double cyc = Cycles == (double)0 ? (double)1 : Cycles;
                double x = cyc * t + Phase;
                double frac = x - math.floor(x);             // [0,1)
                return (double)1 - (double)4 * math.abs(frac - (double)0.5); // -1 at edges, +1 at midpoint
            }
        }
    }
}
