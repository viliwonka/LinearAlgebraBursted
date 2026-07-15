using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for DetMath, the deterministic transcendental surface (OP/DetMath.fProxy.cs). Each function
// is swept over its domain and compared to the Unity.Mathematics oracle within a few-ULP relative+
// absolute tolerance (DetMath is a deterministic polynomial approximation, not bit-identical to
// libm). Total/edge behavior (over/underflow, domain violations, non-finite inputs) is asserted
// explicitly. Everything runs under Burst FloatMode.Default (= Strict).
public class fProxyDetMathTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestsJob : IJob
    {
        public enum T
        {
            Exp, Exp2, Exp10, Log, Log2, Log10, Pow,
            Sin, Cos, SinCos, Tan, Asin, Acos, Atan, Atan2,
            Sinh, Cosh, Tanh, Acosh,
            ExpOverflow, ExpUnderflow, ExpPosInf, ExpNegInf, ExpNaN,
            LogEdges, TrigEdges, AtanEdges, PowEdges,
        }

        public T Type;

        // Type-scaled closeness: absorbs the few-ULP gap between DetMath and libm, and the absolute
        // form (1 + |expected|) keeps zero-crossings (sin/cos/atan) from inflating relative error.
        static void Close(fProxy got, fProxy expected)
        {
            //+skipFor[double]
            const float tol = 1e-5f;
            //-skipFor
            //+emitFor[double]
            //!const double tol = 1e-12;
            //-emitFor
            fProxy diff = math.abs(got - expected);
            fProxy scale = (fProxy)1 + math.abs(expected);
            Assert.IsTrue(diff <= tol * scale);
        }

        public void Execute()
        {
            switch (Type)
            {
                case T.Exp:   Sweep(T.Exp,   (fProxy)(-20), (fProxy)20);  break;
                case T.Exp2:  Sweep(T.Exp2,  (fProxy)(-20), (fProxy)20);  break;
                case T.Exp10: Sweep(T.Exp10, (fProxy)(-8),  (fProxy)8);   break;
                case T.Log:   Sweep(T.Log,   (fProxy)0.05,  (fProxy)100); break;
                case T.Log2:  Sweep(T.Log2,  (fProxy)0.05,  (fProxy)100); break;
                case T.Log10: Sweep(T.Log10, (fProxy)0.05,  (fProxy)100); break;
                case T.Sin:   Sweep(T.Sin,   (fProxy)(-10), (fProxy)10);  break;
                case T.Cos:   Sweep(T.Cos,   (fProxy)(-10), (fProxy)10);  break;
                case T.SinCos:Sweep(T.SinCos,(fProxy)(-10), (fProxy)10);  break;
                case T.Tan:   Sweep(T.Tan,   (fProxy)(-1.3), (fProxy)1.3);break;   // stay off the poles
                case T.Asin:  Sweep(T.Asin,  (fProxy)(-0.99),(fProxy)0.99);break;
                case T.Acos:  Sweep(T.Acos,  (fProxy)(-0.99),(fProxy)0.99);break;
                case T.Atan:  Sweep(T.Atan,  (fProxy)(-100),(fProxy)100); break;
                case T.Sinh:  Sweep(T.Sinh,  (fProxy)(-10), (fProxy)10);  break;
                case T.Cosh:  Sweep(T.Cosh,  (fProxy)(-10), (fProxy)10);  break;
                case T.Tanh:  Sweep(T.Tanh,  (fProxy)(-10), (fProxy)10);  break;
                case T.Acosh: Sweep(T.Acosh, (fProxy)1,     (fProxy)20);  break;
                case T.Pow:   PowSweep();  break;
                case T.Atan2: Atan2Sweep(); break;
                case T.ExpOverflow:  Assert.IsTrue(DetMath.Exp((fProxy)10000) == (fProxy)float.PositiveInfinity); break;
                case T.ExpUnderflow: Assert.IsTrue(DetMath.Exp((fProxy)(-10000)) == (fProxy)0); break;
                case T.ExpPosInf:    Assert.IsTrue(DetMath.Exp((fProxy)float.PositiveInfinity) == (fProxy)float.PositiveInfinity); break;
                case T.ExpNegInf:    Assert.IsTrue(DetMath.Exp((fProxy)float.NegativeInfinity) == (fProxy)0); break;
                case T.ExpNaN:       { fProxy en = DetMath.Exp((fProxy)float.NaN); Assert.IsTrue(en != en); } break;
                case T.LogEdges:  LogEdges();  break;
                case T.TrigEdges: TrigEdges(); break;
                case T.AtanEdges: AtanEdges(); break;
                case T.PowEdges:  PowEdges();  break;
            }
        }

        void Sweep(T k, fProxy lo, fProxy hi)
        {
            const int N = 500;
            for (int i = 0; i <= N; i++)
            {
                fProxy x = lo + (hi - lo) * ((fProxy)i / (fProxy)N);
                switch (k)
                {
                    case T.Exp:   Close(DetMath.Exp(x),   math.exp(x));   break;
                    case T.Exp2:  Close(DetMath.Exp2(x),  math.exp2(x));  break;
                    case T.Exp10: Close(DetMath.Exp10(x), math.exp10(x)); break;
                    case T.Log:   Close(DetMath.Log(x),   math.log(x));   break;
                    case T.Log2:  Close(DetMath.Log2(x),  math.log2(x));  break;
                    case T.Log10: Close(DetMath.Log10(x), math.log10(x)); break;
                    case T.Sin:   Close(DetMath.Sin(x),   math.sin(x));   break;
                    case T.Cos:   Close(DetMath.Cos(x),   math.cos(x));   break;
                    case T.SinCos:
                        DetMath.SinCos(x, out fProxy s, out fProxy c);
                        Close(s, math.sin(x)); Close(c, math.cos(x));
                        break;
                    case T.Tan:   Close(DetMath.Tan(x),   math.tan(x));   break;
                    case T.Asin:  Close(DetMath.Asin(x),  math.asin(x));  break;
                    case T.Acos:  Close(DetMath.Acos(x),  math.acos(x));  break;
                    case T.Atan:  Close(DetMath.Atan(x),  math.atan(x));  break;
                    case T.Sinh:  Close(DetMath.Sinh(x),  math.sinh(x));  break;
                    case T.Cosh:  Close(DetMath.Cosh(x),  math.cosh(x));  break;
                    case T.Tanh:  Close(DetMath.Tanh(x),  math.tanh(x));  break;
                    case T.Acosh: Close(DetMath.Acosh(x), math.log(x + math.sqrt(x * x - (fProxy)1))); break;
                }
            }
        }

        void PowSweep()
        {
            const int N = 60;
            for (int i = 0; i <= N; i++)
            {
                fProxy x = (fProxy)0.1 + (fProxy)9.9 * ((fProxy)i / (fProxy)N);
                for (int j = 0; j <= N; j++)
                {
                    fProxy y = (fProxy)(-3) + (fProxy)6 * ((fProxy)j / (fProxy)N);
                    Close(DetMath.Pow(x, y), math.pow(x, y));
                }
            }
        }

        void Atan2Sweep()
        {
            const int N = 40;
            for (int i = 0; i <= N; i++)
            {
                fProxy y = (fProxy)(-5) + (fProxy)10 * ((fProxy)i / (fProxy)N);
                for (int j = 0; j <= N; j++)
                {
                    fProxy x = (fProxy)(-5) + (fProxy)10 * ((fProxy)j / (fProxy)N);
                    if (math.abs(x) < (fProxy)0.05 && math.abs(y) < (fProxy)0.05) continue;   // skip origin
                    Close(DetMath.Atan2(y, x), math.atan2(y, x));
                }
            }
        }

        void LogEdges()
        {
            fProxy inf = (fProxy)float.PositiveInfinity;
            fProxy ninf = (fProxy)float.NegativeInfinity;
            fProxy nan = (fProxy)float.NaN;
            Assert.IsTrue(DetMath.Log((fProxy)0) == ninf);
            fProxy ln = DetMath.Log((fProxy)(-1)); Assert.IsTrue(ln != ln);
            Assert.IsTrue(DetMath.Log(inf) == inf);
            Assert.IsTrue(math.abs(DetMath.Log((fProxy)1)) <= (fProxy)1e-6);
            fProxy lnan = DetMath.Log(nan); Assert.IsTrue(lnan != lnan);
        }

        void TrigEdges()
        {
            fProxy inf = (fProxy)float.PositiveInfinity;
            fProxy nan = (fProxy)float.NaN;
            fProxy si = DetMath.Sin(inf); Assert.IsTrue(si != si);
            fProxy ci = DetMath.Cos(inf); Assert.IsTrue(ci != ci);
            fProxy sn = DetMath.Sin(nan); Assert.IsTrue(sn != sn);
        }

        void AtanEdges()
        {
            fProxy inf = (fProxy)float.PositiveInfinity;
            fProxy ninf = (fProxy)float.NegativeInfinity;
            Assert.IsTrue(math.abs(DetMath.Atan(inf) - (fProxy)1.5707963267948966) <= (fProxy)1e-5);
            Assert.IsTrue(math.abs(DetMath.Atan(ninf) + (fProxy)1.5707963267948966) <= (fProxy)1e-5);
        }

        void PowEdges()
        {
            Assert.IsTrue(DetMath.Pow((fProxy)0, (fProxy)2) == (fProxy)0);
        }
    }

    static void Run(TestsJob.T t) => new TestsJob { Type = t }.Run();

    [Test] public void Exp()   => Run(TestsJob.T.Exp);
    [Test] public void Exp2()  => Run(TestsJob.T.Exp2);
    [Test] public void Exp10() => Run(TestsJob.T.Exp10);
    [Test] public void Log()   => Run(TestsJob.T.Log);
    [Test] public void Log2()  => Run(TestsJob.T.Log2);
    [Test] public void Log10() => Run(TestsJob.T.Log10);
    [Test] public void Pow()   => Run(TestsJob.T.Pow);
    [Test] public void Sin()   => Run(TestsJob.T.Sin);
    [Test] public void Cos()   => Run(TestsJob.T.Cos);
    [Test] public void SinCos()=> Run(TestsJob.T.SinCos);
    [Test] public void Tan()   => Run(TestsJob.T.Tan);
    [Test] public void Asin()  => Run(TestsJob.T.Asin);
    [Test] public void Acos()  => Run(TestsJob.T.Acos);
    [Test] public void Atan()  => Run(TestsJob.T.Atan);
    [Test] public void Atan2() => Run(TestsJob.T.Atan2);
    [Test] public void Sinh()  => Run(TestsJob.T.Sinh);
    [Test] public void Cosh()  => Run(TestsJob.T.Cosh);
    [Test] public void Tanh()  => Run(TestsJob.T.Tanh);
    [Test] public void Acosh() => Run(TestsJob.T.Acosh);
    [Test] public void ExpOverflow()  => Run(TestsJob.T.ExpOverflow);
    [Test] public void ExpUnderflow() => Run(TestsJob.T.ExpUnderflow);
    [Test] public void ExpPosInf()    => Run(TestsJob.T.ExpPosInf);
    [Test] public void ExpNegInf()    => Run(TestsJob.T.ExpNegInf);
    [Test] public void ExpNaN()       => Run(TestsJob.T.ExpNaN);
    [Test] public void LogEdges()  => Run(TestsJob.T.LogEdges);
    [Test] public void TrigEdges() => Run(TestsJob.T.TrigEdges);
    [Test] public void AtanEdges() => Run(TestsJob.T.AtanEdges);
    [Test] public void PowEdges()  => Run(TestsJob.T.PowEdges);
}
