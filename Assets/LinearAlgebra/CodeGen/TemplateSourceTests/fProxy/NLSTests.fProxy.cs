using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.TestTools;

public class fProxyNLSTests
{
    // ----- Functor structs (Burst-legal: only fProxyN/fProxyMxN fields, no managed state) -----

    // p = [a, b, c]: y = a*exp(-b*x) + c.
    public struct ExpDecayResidual : IfProxyResidualFunction
    {
        public fProxyN X, Y;
        public void Residuals(in fProxyN p, ref fProxyN r)
        {
            for (int i = 0; i < r.N; i++)
                r[i] = p[0] * math.exp(-p[1] * X[i]) + p[2] - Y[i];
        }
    }

    // Same model as ExpDecayResidual, with an analytic Jacobian (for the numeric-vs-analytic check).
    public struct ExpDecayJacobian : IfProxyResidualJacobian
    {
        public fProxyN X, Y;
        public void Residuals(in fProxyN p, ref fProxyN r)
        {
            for (int i = 0; i < r.N; i++)
                r[i] = p[0] * math.exp(-p[1] * X[i]) + p[2] - Y[i];
        }
        public void Jacobian(in fProxyN p, ref fProxyMxN J)
        {
            for (int i = 0; i < J.M_Rows; i++)
            {
                fProxy e = math.exp(-p[1] * X[i]);
                J[i, 0] = e;
                J[i, 1] = -p[0] * X[i] * e;
                J[i, 2] = (fProxy)1;
            }
        }
    }

    // p = [A, w, phi, off]: y = A*sin(w*t + phi) + off.
    public struct SineResidual : IfProxyResidualFunction
    {
        public fProxyN T, Y;
        public void Residuals(in fProxyN p, ref fProxyN r)
        {
            for (int i = 0; i < r.N; i++)
                r[i] = p[0] * math.sin(p[1] * T[i] + p[2]) + p[3] - Y[i];
        }
    }

    // Same model as ExpDecayResidual, plus a 4th parameter (p[3]) the model never references at all
    // (an exactly-zero Jacobian column regardless of p).
    public struct FlatParamResidual : IfProxyResidualFunction
    {
        public fProxyN X, Y;
        public void Residuals(in fProxyN p, ref fProxyN r)
        {
            for (int i = 0; i < r.N; i++)
                r[i] = p[0] * math.exp(-p[1] * X[i]) + p[2] - Y[i];
        }
    }

    // NIST StRD Chwirut2: p = [b1, b2, b3], y = exp(-b1*x) / (b2 + b3*x).
    public struct Chwirut2Residual : IfProxyResidualFunction
    {
        public fProxyN X, Y;
        public void Residuals(in fProxyN p, ref fProxyN r)
        {
            for (int i = 0; i < r.N; i++)
                r[i] = math.exp(-p[0] * X[i]) / (p[1] + p[2] * X[i]) - Y[i];
        }
    }

    // p = [m, b]: y = m*x + b.
    public struct LinearResidual : IfProxyResidualFunction
    {
        public fProxyN X, Y;
        public void Residuals(in fProxyN p, ref fProxyN r)
        {
            for (int i = 0; i < r.N; i++)
                r[i] = p[0] * X[i] + p[1] - Y[i];
        }
    }

    // Curve model for the curveFit facade: y = m*x + b.
    public struct LinModel : IfProxyCurveModel
    {
        public fProxy Eval(fProxy x, in fProxyN p) => p[0] * x + p[1];
    }

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            ExpDecayFit,
            SineFit,
            Chwirut2NIST,
            FlatParameterNoBlowup,
            RobustLossBeatsL2,
            NumericVsAnalyticJacobian,
            CurveFitHappyPath,
            CurveFitWeightedHappyPath,
            RepeatedCallsNoLeak,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff/extra
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ExpDecayFit: ExpDecayFit(); break;
                case TestType.SineFit: SineFit(); break;
                case TestType.Chwirut2NIST: Chwirut2NIST(); break;
                case TestType.FlatParameterNoBlowup: FlatParameterNoBlowup(); break;
                case TestType.RobustLossBeatsL2: RobustLossBeatsL2(); break;
                case TestType.NumericVsAnalyticJacobian: NumericVsAnalyticJacobian(); break;
                case TestType.CurveFitHappyPath: CurveFitHappyPath(); break;
                case TestType.CurveFitWeightedHappyPath: CurveFitWeightedHappyPath(); break;
                case TestType.RepeatedCallsNoLeak: RepeatedCallsNoLeak(); break;
            }
        }

        // 9-point exponential decay: a=2.5, b=1.3, c=0.5, small fixed (non-random) wobble.
        void ExpDecayFit()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = arena.fProxyVec(9);
            var Y = arena.fProxyVec(9);
            SetXY(X, Y, 0, (fProxy)0.0, (fProxy)3.01); SetXY(X, Y, 1, (fProxy)0.5, (fProxy)1.7971);
            SetXY(X, Y, 2, (fProxy)1.0, (fProxy)1.1963); SetXY(X, Y, 3, (fProxy)1.5, (fProxy)0.8507);
            SetXY(X, Y, 4, (fProxy)2.0, (fProxy)0.7057); SetXY(X, Y, 5, (fProxy)2.5, (fProxy)0.5849);
            SetXY(X, Y, 6, (fProxy)3.0, (fProxy)0.5576); SetXY(X, Y, 7, (fProxy)3.5, (fProxy)0.5084);
            SetXY(X, Y, 8, (fProxy)4.0, (fProxy)0.5238);

            var f = new ExpDecayResidual { X = X, Y = Y };
            var p = arena.fProxyVec(3);
            p[0] = (fProxy)1; p[1] = (fProxy)1; p[2] = (fProxy)0;

            var info = Optimize.nlsSolve(ref f, ref p, 9);

            AssertSolved(info);
            AssertFinite(p[0]); AssertFinite(p[1]); AssertFinite(p[2]);
            AssertClose(p[0], (fProxy)2.5, (fProxy)0.02);
            AssertClose(p[1], (fProxy)1.3, (fProxy)0.02);
            AssertClose(p[2], (fProxy)0.5, (fProxy)0.02);

            arena.Dispose();
        }

        // 14-point sine: A=3, w=2, phi=0.4, off=1, small fixed wobble.
        void SineFit()
        {
            var arena = new Arena(Allocator.Persistent);

            var T = arena.fProxyVec(14);
            var Y = arena.fProxyVec(14);
            SetXY(T, Y, 0, (fProxy)0.0, (fProxy)2.1883); SetXY(T, Y, 1, (fProxy)0.5, (fProxy)3.9413);
            SetXY(T, Y, 2, (fProxy)1.0, (fProxy)3.0364); SetXY(T, Y, 3, (fProxy)1.5, (fProxy)0.2134);
            SetXY(T, Y, 4, (fProxy)2.0, (fProxy)(-1.8398)); SetXY(T, Y, 5, (fProxy)2.5, (fProxy)(-1.3283));
            SetXY(T, Y, 6, (fProxy)3.0, (fProxy)1.3696); SetXY(T, Y, 7, (fProxy)3.5, (fProxy)3.6811);
            SetXY(T, Y, 8, (fProxy)4.0, (fProxy)3.5738); SetXY(T, Y, 9, (fProxy)4.5, (fProxy)1.0543);
            SetXY(T, Y, 10, (fProxy)5.0, (fProxy)(-1.4685)); SetXY(T, Y, 11, (fProxy)5.5, (fProxy)(-1.768));
            SetXY(T, Y, 12, (fProxy)6.0, (fProxy)0.5232); SetXY(T, Y, 13, (fProxy)6.5, (fProxy)3.2061);

            var f = new SineResidual { T = T, Y = Y };
            var p = arena.fProxyVec(4);
            p[0] = (fProxy)1; p[1] = (fProxy)1.8; p[2] = (fProxy)0; p[3] = (fProxy)0;

            var info = Optimize.nlsSolve(ref f, ref p, 14, Consts.fProxySqrtEps, Consts.fProxyEpsilon, 500, NLSJacobianMode.Forward, (fProxy)0);

            AssertSolved(info);
            AssertFinite(p[0]); AssertFinite(p[1]); AssertFinite(p[2]); AssertFinite(p[3]);
            AssertClose(p[0], (fProxy)3.0, (fProxy)0.02);
            AssertClose(p[1], (fProxy)2.0, (fProxy)0.02);
            AssertClose(p[2], (fProxy)0.4, (fProxy)0.02);
            AssertClose(p[3], (fProxy)1.0, (fProxy)0.02);

            arena.Dispose();
        }

        // NIST StRD Chwirut2 (54 observations, https://www.itl.nist.gov/div898/strd/nls/data/chwirut2.shtml):
        // y = exp(-b1*x)/(b2+b3*x). Certified: b1=1.6657666537E-1, b2=5.1653291286E-3, b3=1.2150007096E-2.
        // Start1 = (0.1, 0.01, 0.02) (NIST-prescribed).
        void Chwirut2NIST()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = arena.fProxyVec(54);
            var Y = arena.fProxyVec(54);
            SetXY(X, Y, 0, (fProxy)0.5, (fProxy)92.9); SetXY(X, Y, 1, (fProxy)1.0, (fProxy)57.1); SetXY(X, Y, 2, (fProxy)1.75, (fProxy)31.05);
            SetXY(X, Y, 3, (fProxy)3.75, (fProxy)11.5875); SetXY(X, Y, 4, (fProxy)5.75, (fProxy)8.025); SetXY(X, Y, 5, (fProxy)0.875, (fProxy)63.6);
            SetXY(X, Y, 6, (fProxy)2.25, (fProxy)21.4); SetXY(X, Y, 7, (fProxy)3.25, (fProxy)14.25); SetXY(X, Y, 8, (fProxy)5.25, (fProxy)8.475);
            SetXY(X, Y, 9, (fProxy)0.75, (fProxy)63.8); SetXY(X, Y, 10, (fProxy)1.75, (fProxy)26.8); SetXY(X, Y, 11, (fProxy)2.75, (fProxy)16.4625);
            SetXY(X, Y, 12, (fProxy)4.75, (fProxy)7.125); SetXY(X, Y, 13, (fProxy)0.625, (fProxy)67.3); SetXY(X, Y, 14, (fProxy)1.25, (fProxy)41.0);
            SetXY(X, Y, 15, (fProxy)2.25, (fProxy)21.15); SetXY(X, Y, 16, (fProxy)4.25, (fProxy)8.175); SetXY(X, Y, 17, (fProxy)0.5, (fProxy)81.5);
            SetXY(X, Y, 18, (fProxy)3.0, (fProxy)13.12); SetXY(X, Y, 19, (fProxy)0.75, (fProxy)59.9); SetXY(X, Y, 20, (fProxy)3.0, (fProxy)14.62);
            SetXY(X, Y, 21, (fProxy)1.5, (fProxy)32.9); SetXY(X, Y, 22, (fProxy)6.0, (fProxy)5.44); SetXY(X, Y, 23, (fProxy)3.0, (fProxy)12.56);
            SetXY(X, Y, 24, (fProxy)6.0, (fProxy)5.44); SetXY(X, Y, 25, (fProxy)1.5, (fProxy)32.0); SetXY(X, Y, 26, (fProxy)3.0, (fProxy)13.95);
            SetXY(X, Y, 27, (fProxy)0.5, (fProxy)75.8); SetXY(X, Y, 28, (fProxy)2.0, (fProxy)20.0); SetXY(X, Y, 29, (fProxy)4.0, (fProxy)10.42);
            SetXY(X, Y, 30, (fProxy)0.75, (fProxy)59.5); SetXY(X, Y, 31, (fProxy)2.0, (fProxy)21.67); SetXY(X, Y, 32, (fProxy)5.0, (fProxy)8.55);
            SetXY(X, Y, 33, (fProxy)0.75, (fProxy)62.0); SetXY(X, Y, 34, (fProxy)2.25, (fProxy)20.2); SetXY(X, Y, 35, (fProxy)3.75, (fProxy)7.76);
            SetXY(X, Y, 36, (fProxy)5.75, (fProxy)3.75); SetXY(X, Y, 37, (fProxy)3.0, (fProxy)11.81); SetXY(X, Y, 38, (fProxy)0.75, (fProxy)54.7);
            SetXY(X, Y, 39, (fProxy)2.5, (fProxy)23.7); SetXY(X, Y, 40, (fProxy)4.0, (fProxy)11.55); SetXY(X, Y, 41, (fProxy)0.75, (fProxy)61.3);
            SetXY(X, Y, 42, (fProxy)2.5, (fProxy)17.7); SetXY(X, Y, 43, (fProxy)4.0, (fProxy)8.74); SetXY(X, Y, 44, (fProxy)0.75, (fProxy)59.2);
            SetXY(X, Y, 45, (fProxy)2.5, (fProxy)16.3); SetXY(X, Y, 46, (fProxy)4.0, (fProxy)8.62); SetXY(X, Y, 47, (fProxy)0.5, (fProxy)81.0);
            SetXY(X, Y, 48, (fProxy)6.0, (fProxy)4.87); SetXY(X, Y, 49, (fProxy)3.0, (fProxy)14.62); SetXY(X, Y, 50, (fProxy)0.5, (fProxy)81.7);
            SetXY(X, Y, 51, (fProxy)2.75, (fProxy)17.17); SetXY(X, Y, 52, (fProxy)0.5, (fProxy)81.3); SetXY(X, Y, 53, (fProxy)1.75, (fProxy)28.9);

            var f = new Chwirut2Residual { X = X, Y = Y };
            var p = arena.fProxyVec(3);
            p[0] = (fProxy)0.1; p[1] = (fProxy)0.01; p[2] = (fProxy)0.02;

            var info = Optimize.nlsSolve(ref f, ref p, 54);

            AssertSolved(info);
            AssertFinite(p[0]); AssertFinite(p[1]); AssertFinite(p[2]);
            AssertClose(p[0], (fProxy)1.6657666537e-1, (fProxy)5e-3);
            AssertClose(p[1], (fProxy)5.1653291286e-3, (fProxy)5e-3);
            AssertClose(p[2], (fProxy)1.2150007096e-2, (fProxy)5e-3);

            arena.Dispose();
        }

        // Reuses the ExpDecayFit dataset with an extra unused 4th parameter: its Jacobian column is
        // exactly zero for every p, so it must stay EXACTLY at its starting value (never move, never
        // blow up) regardless of that value's magnitude, while a/b/c converge as in ExpDecayFit.
        void FlatParameterNoBlowup()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = arena.fProxyVec(9);
            var Y = arena.fProxyVec(9);
            SetXY(X, Y, 0, (fProxy)0.0, (fProxy)3.01); SetXY(X, Y, 1, (fProxy)0.5, (fProxy)1.7971);
            SetXY(X, Y, 2, (fProxy)1.0, (fProxy)1.1963); SetXY(X, Y, 3, (fProxy)1.5, (fProxy)0.8507);
            SetXY(X, Y, 4, (fProxy)2.0, (fProxy)0.7057); SetXY(X, Y, 5, (fProxy)2.5, (fProxy)0.5849);
            SetXY(X, Y, 6, (fProxy)3.0, (fProxy)0.5576); SetXY(X, Y, 7, (fProxy)3.5, (fProxy)0.5084);
            SetXY(X, Y, 8, (fProxy)4.0, (fProxy)0.5238);

            var f = new FlatParamResidual { X = X, Y = Y };

            var p1 = arena.fProxyVec(4);
            p1[0] = (fProxy)1; p1[1] = (fProxy)1; p1[2] = (fProxy)0; p1[3] = (fProxy)0;
            var info1 = Optimize.nlsSolve(ref f, ref p1, 9);
            AssertSolved(info1);
            AssertFinite(p1[0]); AssertFinite(p1[1]); AssertFinite(p1[2]); AssertFinite(p1[3]);
            AssertClose(p1[0], (fProxy)2.5, (fProxy)0.02);
            AssertClose(p1[1], (fProxy)1.3, (fProxy)0.02);
            AssertClose(p1[2], (fProxy)0.5, (fProxy)0.02);
            AssertClose(p1[3], (fProxy)0, (fProxy)0); // untouched: exactly its starting value

            var p2 = arena.fProxyVec(4);
            p2[0] = (fProxy)1; p2[1] = (fProxy)1; p2[2] = (fProxy)0; p2[3] = (fProxy)(-1000000);
            var info2 = Optimize.nlsSolve(ref f, ref p2, 9);
            AssertSolved(info2);
            AssertFinite(p2[0]); AssertFinite(p2[1]); AssertFinite(p2[2]); AssertFinite(p2[3]);
            AssertClose(p2[0], (fProxy)2.5, (fProxy)0.02);
            AssertClose(p2[3], (fProxy)(-1000000), (fProxy)0); // untouched, even from a wild start

            arena.Dispose();
        }

        // 12-point line (m=2, b=-1) with 2 gross outliers (magnitude ~20-30 on a 0-22 range): plain
        // L2 is visibly pulled off the true line, Huber/Tukey both recover it.
        void RobustLossBeatsL2()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = arena.fProxyVec(12);
            var Y = arena.fProxyVec(12);
            SetXY(X, Y, 0, (fProxy)0, (fProxy)(-0.95)); SetXY(X, Y, 1, (fProxy)1, (fProxy)0.97);
            SetXY(X, Y, 2, (fProxy)2, (fProxy)3.02); SetXY(X, Y, 3, (fProxy)3, (fProxy)29.96); // outlier
            SetXY(X, Y, 4, (fProxy)4, (fProxy)7.03); SetXY(X, Y, 5, (fProxy)5, (fProxy)8.98);
            SetXY(X, Y, 6, (fProxy)6, (fProxy)11.04); SetXY(X, Y, 7, (fProxy)7, (fProxy)12.97);
            SetXY(X, Y, 8, (fProxy)8, (fProxy)15.02); SetXY(X, Y, 9, (fProxy)9, (fProxy)(-3.05)); // outlier
            SetXY(X, Y, 10, (fProxy)10, (fProxy)19.03); SetXY(X, Y, 11, (fProxy)11, (fProxy)20.98);

            fProxy mTrue = (fProxy)2, bTrue = (fProxy)(-1);

            var fL2 = new LinearResidual { X = X, Y = Y };
            var pL2 = arena.fProxyVec(2);
            pL2[0] = (fProxy)1.5; pL2[1] = (fProxy)(-0.5);
            var infoL2 = Optimize.nlsSolve(ref fL2, ref pL2, 12);
            AssertSolved(infoL2);
            fProxy relL2 = RelErr2(pL2[0], pL2[1], mTrue, bTrue);

            var fHuber = new LinearResidual { X = X, Y = Y };
            var pHuber = arena.fProxyVec(2);
            pHuber[0] = (fProxy)1.5; pHuber[1] = (fProxy)(-0.5);
            var huberLoss = new fProxyHuberLoss((fProxy)0.3);
            var infoHuber = Optimize.nlsSolve(ref fHuber, ref pHuber, 12, in huberLoss);
            AssertSolved(infoHuber);
            fProxy relHuber = RelErr2(pHuber[0], pHuber[1], mTrue, bTrue);

            var fTukey = new LinearResidual { X = X, Y = Y };
            var pTukey = arena.fProxyVec(2);
            pTukey[0] = (fProxy)1.5; pTukey[1] = (fProxy)(-0.5);
            var tukeyLoss = new fProxyTukeyLoss((fProxy)4.685);
            var infoTukey = Optimize.nlsSolve(ref fTukey, ref pTukey, 12, in tukeyLoss);
            AssertSolved(infoTukey);
            fProxy relTukey = RelErr2(pTukey[0], pTukey[1], mTrue, bTrue);

            AssertFinite(relL2); AssertFinite(relHuber); AssertFinite(relTukey);

            // L2 is visibly wrong; both robust losses recover the true line.
            AssertGreater(relL2, (fProxy)0.5);
            AssertLess(relHuber, (fProxy)0.1);
            AssertLess(relTukey, (fProxy)0.05);
            AssertLess(relHuber, relL2);
            AssertLess(relTukey, relL2);

            arena.Dispose();
        }

        // Numeric (forward AND central) vs analytic Jacobian on the ExpDecayFit dataset: all three
        // must reach essentially the same optimum.
        void NumericVsAnalyticJacobian()
        {
            var arena = new Arena(Allocator.Persistent);

            var X = arena.fProxyVec(9);
            var Y = arena.fProxyVec(9);
            SetXY(X, Y, 0, (fProxy)0.0, (fProxy)3.01); SetXY(X, Y, 1, (fProxy)0.5, (fProxy)1.7971);
            SetXY(X, Y, 2, (fProxy)1.0, (fProxy)1.1963); SetXY(X, Y, 3, (fProxy)1.5, (fProxy)0.8507);
            SetXY(X, Y, 4, (fProxy)2.0, (fProxy)0.7057); SetXY(X, Y, 5, (fProxy)2.5, (fProxy)0.5849);
            SetXY(X, Y, 6, (fProxy)3.0, (fProxy)0.5576); SetXY(X, Y, 7, (fProxy)3.5, (fProxy)0.5084);
            SetXY(X, Y, 8, (fProxy)4.0, (fProxy)0.5238);

            var fNum = new ExpDecayResidual { X = X, Y = Y };
            var pNum = arena.fProxyVec(3);
            pNum[0] = (fProxy)1; pNum[1] = (fProxy)1; pNum[2] = (fProxy)0;
            var infoNum = Optimize.nlsSolve(ref fNum, ref pNum, 9, Consts.fProxySqrtEps, Consts.fProxyEpsilon, 200, NLSJacobianMode.Forward, (fProxy)0);
            AssertSolved(infoNum);

            var fCentral = new ExpDecayResidual { X = X, Y = Y };
            var pCentral = arena.fProxyVec(3);
            pCentral[0] = (fProxy)1; pCentral[1] = (fProxy)1; pCentral[2] = (fProxy)0;
            var infoCentral = Optimize.nlsSolve(ref fCentral, ref pCentral, 9, Consts.fProxySqrtEps, Consts.fProxyEpsilon, 200, NLSJacobianMode.Central, (fProxy)0);
            AssertSolved(infoCentral);

            var fAna = new ExpDecayJacobian { X = X, Y = Y };
            var pAna = arena.fProxyVec(3);
            pAna[0] = (fProxy)1; pAna[1] = (fProxy)1; pAna[2] = (fProxy)0;
            var infoAna = Optimize.nlsSolve(ref fAna, ref pAna, 9, Consts.fProxySqrtEps, Consts.fProxyEpsilon, 200);
            AssertSolved(infoAna);

            AssertClose(pNum[0], pAna[0], (fProxy)1e-4);
            AssertClose(pNum[1], pAna[1], (fProxy)1e-4);
            AssertClose(pNum[2], pAna[2], (fProxy)1e-4);
            AssertClose(pCentral[0], pAna[0], (fProxy)1e-4);
            AssertClose(pCentral[1], pAna[1], (fProxy)1e-4);
            AssertClose(pCentral[2], pAna[2], (fProxy)1e-4);

            arena.Dispose();
        }

        // curveFit facade happy path: y = 2x + 1, tiny fixed wobble.
        void CurveFitHappyPath()
        {
            var arena = new Arena(Allocator.Persistent);

            var xdata = arena.fProxyVec(5);
            var ydata = arena.fProxyVec(5);
            SetXY(xdata, ydata, 0, (fProxy)0, (fProxy)1.01); SetXY(xdata, ydata, 1, (fProxy)1, (fProxy)2.98);
            SetXY(xdata, ydata, 2, (fProxy)2, (fProxy)5.015); SetXY(xdata, ydata, 3, (fProxy)3, (fProxy)6.99);
            SetXY(xdata, ydata, 4, (fProxy)4, (fProxy)9.005);

            var model = new LinModel();
            var p = arena.fProxyVec(2);
            p[0] = (fProxy)0; p[1] = (fProxy)0;

            var info = Optimize.curveFit(in xdata, in ydata, ref model, ref p);

            AssertSolved(info);
            AssertFinite(p[0]); AssertFinite(p[1]);
            AssertClose(p[0], (fProxy)2, (fProxy)0.01);
            AssertClose(p[1], (fProxy)1, (fProxy)0.01);

            arena.Dispose();
        }

        // Weighted curveFit (uniform sigma == unweighted) reproduces CurveFitHappyPath's result.
        void CurveFitWeightedHappyPath()
        {
            var arena = new Arena(Allocator.Persistent);

            var xdata = arena.fProxyVec(5);
            var ydata = arena.fProxyVec(5);
            SetXY(xdata, ydata, 0, (fProxy)0, (fProxy)1.01); SetXY(xdata, ydata, 1, (fProxy)1, (fProxy)2.98);
            SetXY(xdata, ydata, 2, (fProxy)2, (fProxy)5.015); SetXY(xdata, ydata, 3, (fProxy)3, (fProxy)6.99);
            SetXY(xdata, ydata, 4, (fProxy)4, (fProxy)9.005);

            var sigma = arena.fProxyVec(5);
            for (int i = 0; i < 5; i++) sigma[i] = (fProxy)1;

            var model = new LinModel();
            var p = arena.fProxyVec(2);
            p[0] = (fProxy)0; p[1] = (fProxy)0;

            var info = Optimize.curveFit(in xdata, in ydata, in sigma, ref model, ref p);

            AssertSolved(info);
            AssertClose(p[0], (fProxy)2, (fProxy)0.01);
            AssertClose(p[1], (fProxy)1, (fProxy)0.01);

            arena.Dispose();
        }

        // Calls curveFit 30 times in a row on the same small dataset: repeated-call state
        // stability, not leak detection (the scratch is Allocator.Temp, whose leaks are not
        // reliably caught by Unity's collections checks inside a Burst job).
        void RepeatedCallsNoLeak()
        {
            var arena = new Arena(Allocator.Persistent);

            var xdata = arena.fProxyVec(5);
            var ydata = arena.fProxyVec(5);
            SetXY(xdata, ydata, 0, (fProxy)0, (fProxy)1.01); SetXY(xdata, ydata, 1, (fProxy)1, (fProxy)2.98);
            SetXY(xdata, ydata, 2, (fProxy)2, (fProxy)5.015); SetXY(xdata, ydata, 3, (fProxy)3, (fProxy)6.99);
            SetXY(xdata, ydata, 4, (fProxy)4, (fProxy)9.005);

            var model = new LinModel();
            var p = arena.fProxyVec(2);

            for (int rep = 0; rep < 30; rep++)
            {
                p[0] = (fProxy)0; p[1] = (fProxy)0;
                var info = Optimize.curveFit(in xdata, in ydata, ref model, ref p);
                AssertSolved(info);
                AssertClose(p[0], (fProxy)2, (fProxy)0.01);
                AssertClose(p[1], (fProxy)1, (fProxy)0.01);
            }

            arena.Dispose();
        }

        static void SetXY(fProxyN X, fProxyN Y, int i, fProxy x, fProxy y)
        {
            X[i] = x;
            Y[i] = y;
        }

        static fProxy RelErr2(fProxy a0, fProxy a1, fProxy t0, fProxy t1)
        {
            fProxy dx0 = a0 - t0, dx1 = a1 - t1;
            fProxy num = math.sqrt(dx0 * dx0 + dx1 * dx1);
            fProxy den = math.sqrt(t0 * t0 + t1 * t1);
            return num / den;
        }

        private void AssertSolved(NLSInfo info)
        {
            bool ok = info.Solved;
            if (!ok && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = (fProxy)(int)info.status;
                Fail[2] = (fProxy)0;
                Fail[3] = (fProxy)0;
            }
            Assert.IsTrue(ok);
        }

        private void AssertFinite(fProxy v)
        {
            if (!math.isfinite(v) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = v;
                Fail[2] = (fProxy)0;
                Fail[3] = (fProxy)0;
            }
            Assert.IsTrue(math.isfinite(v));
        }

        private void AssertClose(fProxy a, fProxy b, fProxy precision)
        {
            fProxy diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }

        private void AssertLess(fProxy a, fProxy b)
        {
            if (!(a < b) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = (fProxy)0;
            }
            Assert.IsTrue(a < b);
        }

        private void AssertGreater(fProxy a, fProxy b)
        {
            if (!(a > b) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = (fProxy)0;
            }
            Assert.IsTrue(a > b);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void NLSTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // Managed throw-tests: argument validation runs on the main thread (not in a Burst job).

    [Test]
    public void CurveFitThrowsOnDimMismatch()
    {
        var arena = new Arena(Allocator.Persistent);

        var xdata = arena.fProxyVec(3);
        var ydata = arena.fProxyVec(4);
        var model = new LinModel();
        var p = arena.fProxyVec(2);

        Assert.Catch<ArgumentException>(() =>
            Optimize.curveFit(in xdata, in ydata, ref model, ref p));

        arena.Dispose();
    }

    [Test]
    public void CurveFitWeightedThrowsOnSigmaMismatch()
    {
        var arena = new Arena(Allocator.Persistent);

        var xdata = arena.fProxyVec(3);
        var ydata = arena.fProxyVec(3);
        var sigma = arena.fProxyVec(4);
        var model = new LinModel();
        var p = arena.fProxyVec(2);

        Assert.Catch<ArgumentException>(() =>
            Optimize.curveFit(in xdata, in ydata, in sigma, ref model, ref p));

        arena.Dispose();
    }

    [Test]
    public void NlsSolveThrowsOnMaxIterZero()
    {
        var arena = new Arena(Allocator.Persistent);

        var X = arena.fProxyVec(3);
        var Y = arena.fProxyVec(3);
        var f = new ExpDecayResidual { X = X, Y = Y };
        var p = arena.fProxyVec(3);

        Assert.Catch<ArgumentException>(() =>
            Optimize.nlsSolve(ref f, ref p, 3, Consts.fProxySqrtEps, Consts.fProxyEpsilon, 0, NLSJacobianMode.Forward, (fProxy)0));

        arena.Dispose();
    }

    [Test]
    public void NlsSolveThrowsOnNonPositiveM()
    {
        var arena = new Arena(Allocator.Persistent);

        var X = arena.fProxyVec(3);
        var Y = arena.fProxyVec(3);
        var f = new ExpDecayResidual { X = X, Y = Y };
        var p = arena.fProxyVec(3);

        Assert.Catch<ArgumentException>(() => Optimize.nlsSolve(ref f, ref p, 0));

        arena.Dispose();
    }

    [Test]
    public void HuberLossThrowsOnNonPositiveScale()
    {
        Assert.Catch<ArgumentException>(() => new fProxyHuberLoss((fProxy)0));
        Assert.Catch<ArgumentException>(() => new fProxyHuberLoss((fProxy)(-1)));
    }

    [Test]
    public void TukeyLossThrowsOnNonPositiveScale()
    {
        Assert.Catch<ArgumentException>(() => new fProxyTukeyLoss((fProxy)0));
        Assert.Catch<ArgumentException>(() => new fProxyTukeyLoss((fProxy)(-1)));
    }
}
