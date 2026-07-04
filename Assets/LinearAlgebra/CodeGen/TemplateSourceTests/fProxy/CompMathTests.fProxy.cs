using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for the fProxyComp elementwise math surface (OP.Component.fProxy.cs forwarding to
// UnsafeMathOP). Every case uses Unity.Mathematics math.xxx on the SAME (pre-mutation) inputs as
// the oracle, so the expected value is computed by the exact function the kernel uses - the checks
// are therefore near-exact (a small type-scaled relative tolerance absorbs only reassociation).
//
// The unary math ops are batched: each is one enum value dispatched through a single Unary() body
// (input builder + apply switch + oracle switch), rather than 27 copy-paste methods. Binary/ternary
// ops get their own methods because each has a distinct mutation contract (which buffer is written,
// which must stay unchanged) that is itself under test - these are the classic forwarder failure
// modes (argument order, wrong mutation target).
public class fProxyCompMathTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestsJob : IJob
    {
        public enum TestType
        {
            // ---- unary (oracle = the exact math.xxx the kernel calls) ----
            Abs, Sign, Sqrt, Rsqrt, Acos, Asin, Atan, Acosh,
            Ceil, Floor, Round, Cos, Cosh, Sin, Sinh, Tan, Tanh,
            Exp, Exp2, Exp10, Log, Log2, Log10, Saturate, Frac, Rcp, Relu,

            // ---- pow (int exponent) ----
            PowExponents,

            // ---- interpolation / edges / fused ----
            Lerp, Unlerp, SmoothstepBuffers, SmoothstepScalarEdges, Step,
            Mad, Remap, DegreesRadians, Atan2, MinBuf, MaxBuf, Sincos,
            AbsDiff, SqrDiff,

            // ---- matrix path (generic over IUnsafefProxyArray) ----
            MatrixAbs, MatrixExp, MatrixMad,

            // ---- degenerate shapes ----
            SingleElement, EmptyBuffer,
        }

        public TestType Type;

        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);
            try
            {
                switch (Type)
                {
                    // unary ops all share one body
                    case TestType.Abs: case TestType.Sign: case TestType.Sqrt:
                    case TestType.Rsqrt: case TestType.Acos: case TestType.Asin:
                    case TestType.Atan: case TestType.Acosh: case TestType.Ceil:
                    case TestType.Floor: case TestType.Round: case TestType.Cos:
                    case TestType.Cosh: case TestType.Sin: case TestType.Sinh:
                    case TestType.Tan: case TestType.Tanh: case TestType.Exp:
                    case TestType.Exp2: case TestType.Exp10: case TestType.Log:
                    case TestType.Log2: case TestType.Log10: case TestType.Saturate:
                    case TestType.Frac: case TestType.Rcp: case TestType.Relu:
                        Unary(Type, ref arena);
                        break;

                    case TestType.PowExponents: PowExponents(ref arena); break;

                    case TestType.Lerp: LerpTest(ref arena); break;
                    case TestType.Unlerp: UnlerpTest(ref arena); break;
                    case TestType.SmoothstepBuffers: SmoothstepBuffersTest(ref arena); break;
                    case TestType.SmoothstepScalarEdges: SmoothstepScalarEdgesTest(ref arena); break;
                    case TestType.Step: StepTest(ref arena); break;
                    case TestType.Mad: MadTest(ref arena); break;
                    case TestType.Remap: RemapTest(ref arena); break;
                    case TestType.DegreesRadians: DegreesRadiansTest(ref arena); break;
                    case TestType.Atan2: Atan2Test(ref arena); break;
                    case TestType.MinBuf: MinBufTest(ref arena); break;
                    case TestType.MaxBuf: MaxBufTest(ref arena); break;
                    case TestType.Sincos: SincosTest(ref arena); break;
                    case TestType.AbsDiff: AbsDiffTest(ref arena); break;
                    case TestType.SqrDiff: SqrDiffTest(ref arena); break;

                    case TestType.MatrixAbs: MatrixAbsTest(ref arena); break;
                    case TestType.MatrixExp: MatrixExpTest(ref arena); break;
                    case TestType.MatrixMad: MatrixMadTest(ref arena); break;

                    case TestType.SingleElement: SingleElementTest(ref arena); break;
                    case TestType.EmptyBuffer: EmptyBufferTest(ref arena); break;

                    default: throw new NotImplementedException();
                }
            }
            finally
            {
                arena.Dispose();
            }
        }

        // true only when fProxy expands to double (doubleEpsilon ~2.2e-16 < 1e-10).
        private bool IsDouble() => (double)Consts.fProxyEpsilon < 1e-10;

        // Type-scaled relative+absolute closeness. Oracle uses the same math.xxx as the kernel, so
        // the gap is essentially reassociation-only; float gets a looser band than double.
        private void AssertClose(fProxy got, fProxy expected)
        {
            fProxy tol = IsDouble() ? (fProxy)1e-11 : (fProxy)1e-4;
            fProxy diff = math.abs(got - expected);
            fProxy scale = (fProxy)1 + math.abs(expected);
            Assert.IsTrue(diff <= tol * scale);
        }

        // Domain-safe deterministic input per unary op (sqrt/log need x>0, acos/asin need [-1,1],
        // acosh needs x>=1, tan avoids +/-pi/2, exp10 kept small so 10^x doesn't overflow).
        private fProxyN MakeUnaryInput(ref Arena arena, TestType k, int n)
        {
            switch (k)
            {
                case TestType.Sqrt:
                case TestType.Log:
                case TestType.Log2:
                case TestType.Log10:
                    return arena.fProxyLinVec(n, (fProxy)0.02, (fProxy)9);

                case TestType.Rsqrt:
                case TestType.Rcp:
                    return arena.fProxyLinVec(n, (fProxy)0.05, (fProxy)6);

                case TestType.Acos:
                case TestType.Asin:
                    return arena.fProxyLinVec(n, (fProxy)(-0.95), (fProxy)0.95);

                case TestType.Acosh:
                    return arena.fProxyLinVec(n, (fProxy)1, (fProxy)4);

                case TestType.Tan:
                    return arena.fProxyLinVec(n, (fProxy)(-1.2), (fProxy)1.2);

                case TestType.Exp10:
                    return arena.fProxyLinVec(n, (fProxy)(-2), (fProxy)3);

                case TestType.Exp:
                case TestType.Exp2:
                    return arena.fProxyLinVec(n, (fProxy)(-3), (fProxy)4);

                case TestType.Abs:
                case TestType.Sign:
                case TestType.Relu:
                {
                    // span negatives .. positives and pin an exact zero (sign/relu boundary).
                    var v = arena.fProxyLinVec(n, (fProxy)(-3), (fProxy)3);
                    v[0] = (fProxy)0;
                    return v;
                }

                default:
                    // ceil/floor/round/cos/cosh/sin/sinh/tanh/atan/saturate/frac: any real ok.
                    return arena.fProxyLinVec(n, (fProxy)(-3.3), (fProxy)3.7);
            }
        }

        private void Unary(TestType k, ref Arena arena)
        {
            int n = 37; // odd, not a SIMD multiple -> exercises the scalar tail
            fProxyN x = MakeUnaryInput(ref arena, k, n);
            fProxyN orig = x.Copy();

            switch (k)
            {
                case TestType.Abs: x.absInPlace(); break;
                case TestType.Sign: x.signInPlace(); break;
                case TestType.Sqrt: x.sqrtInPlace(); break;
                case TestType.Rsqrt: x.rsqrtInPlace(); break;
                case TestType.Acos: x.acosInPlace(); break;
                case TestType.Asin: x.asinInPlace(); break;
                case TestType.Atan: x.atanInPlace(); break;
                case TestType.Acosh: x.acoshInPlace(); break;
                case TestType.Ceil: x.ceilInPlace(); break;
                case TestType.Floor: x.floorInPlace(); break;
                case TestType.Round: x.roundInPlace(); break;
                case TestType.Cos: x.cosInPlace(); break;
                case TestType.Cosh: x.coshInPlace(); break;
                case TestType.Sin: x.sinInPlace(); break;
                case TestType.Sinh: x.sinhInPlace(); break;
                case TestType.Tan: x.tanInPlace(); break;
                case TestType.Tanh: x.tanhInPlace(); break;
                case TestType.Exp: x.expInPlace(); break;
                case TestType.Exp2: x.exp2InPlace(); break;
                case TestType.Exp10: x.exp10InPlace(); break;
                case TestType.Log: x.logInPlace(); break;
                case TestType.Log2: x.log2InPlace(); break;
                case TestType.Log10: x.log10InPlace(); break;
                case TestType.Saturate: x.saturateInPlace(); break;
                case TestType.Frac: x.fracInPlace(); break;
                case TestType.Rcp: x.rcpInPlace(); break;
                case TestType.Relu: x.reluInPlace(); break;
                default: throw new NotImplementedException();
            }

            for (int i = 0; i < n; i++)
            {
                fProxy o = orig[i];
                fProxy expected;
                switch (k)
                {
                    case TestType.Abs: expected = math.abs(o); break;
                    case TestType.Sign: expected = math.sign(o); break;
                    case TestType.Sqrt: expected = math.sqrt(o); break;
                    case TestType.Rsqrt: expected = math.rsqrt(o); break;
                    case TestType.Acos: expected = math.acos(o); break;
                    case TestType.Asin: expected = math.asin(o); break;
                    case TestType.Atan: expected = math.atan(o); break;
                    // no math.acosh: kernel is log(x + sqrt(x^2 - 1)), domain x>=1.
                    case TestType.Acosh: expected = math.log(o + math.sqrt(o * o - (fProxy)1)); break;
                    case TestType.Ceil: expected = math.ceil(o); break;
                    case TestType.Floor: expected = math.floor(o); break;
                    case TestType.Round: expected = math.round(o); break;
                    case TestType.Cos: expected = math.cos(o); break;
                    case TestType.Cosh: expected = math.cosh(o); break;
                    case TestType.Sin: expected = math.sin(o); break;
                    case TestType.Sinh: expected = math.sinh(o); break;
                    case TestType.Tan: expected = math.tan(o); break;
                    case TestType.Tanh: expected = math.tanh(o); break;
                    case TestType.Exp: expected = math.exp(o); break;
                    case TestType.Exp2: expected = math.exp2(o); break;
                    // kernel uses math.pow(10, x), not a math.exp10 (which does not exist).
                    case TestType.Exp10: expected = math.pow((fProxy)10, o); break;
                    case TestType.Log: expected = math.log(o); break;
                    case TestType.Log2: expected = math.log2(o); break;
                    case TestType.Log10: expected = math.log10(o); break;
                    case TestType.Saturate: expected = math.saturate(o); break;
                    case TestType.Frac: expected = o - math.floor(o); break;
                    case TestType.Rcp: expected = math.rcp(o); break;
                    case TestType.Relu: expected = o < (fProxy)0 ? (fProxy)0 : o; break;
                    default: throw new NotImplementedException();
                }
                AssertClose(x[i], expected);
            }
        }

        // ---- pow: int exponent (0,1,2,3 and negatives). Positive bases keep math.pow well-defined
        //      for the reciprocal-power (negative-exponent) cases. ----
        private void PowExponents(ref Arena arena)
        {
            CheckPow(ref arena, 0);
            CheckPow(ref arena, 1);
            CheckPow(ref arena, 2);
            CheckPow(ref arena, 3);
            CheckPow(ref arena, -1);
            CheckPow(ref arena, -2);
        }

        private void CheckPow(ref Arena arena, int e)
        {
            int n = 16;
            fProxyN x = arena.fProxyLinVec(n, (fProxy)0.3, (fProxy)3.5);
            fProxyN orig = x.Copy();
            x.powInPlace(e);
            for (int i = 0; i < n; i++)
                AssertClose(x[i], math.pow(orig[i], (fProxy)e)); // exponent 0 -> 1, negative -> reciprocal power
        }

        // ---- interpolation / edges / fused ----

        private void LerpTest(ref Arena arena)
        {
            int n = 24;
            fProxyN a = arena.fProxyLinVec(n, (fProxy)(-2), (fProxy)5);
            fProxyN b = arena.fProxyLinVec(n, (fProxy)3, (fProxy)9);
            fProxyN a0 = a.Copy();
            fProxyN b0 = b.Copy();
            fProxy t = (fProxy)0.35;

            a.lerpInPlace(b, t); // a[i] = lerp(a[i], b[i], t); a mutated, b untouched

            for (int i = 0; i < n; i++)
            {
                AssertClose(a[i], math.lerp(a0[i], b0[i], t));
                AssertClose(b[i], b0[i]);
            }
        }

        private void UnlerpTest(ref Arena arena)
        {
            int n = 24;
            fProxyN a = arena.fProxyLinVec(n, (fProxy)(-2), (fProxy)1);
            fProxyN b = arena.fProxyLinVec(n, (fProxy)6, (fProxy)12); // b != a everywhere (no /0)
            fProxyN a0 = a.Copy();
            fProxyN b0 = b.Copy();
            fProxy t = (fProxy)0.4;

            a.unlerpInPlace(b, t);

            for (int i = 0; i < n; i++)
            {
                AssertClose(a[i], math.unlerp(a0[i], b0[i], t));
                AssertClose(b[i], b0[i]);
            }
        }

        private void SmoothstepBuffersTest(ref Arena arena)
        {
            int n = 24;
            fProxyN a = arena.fProxyLinVec(n, (fProxy)(-1), (fProxy)1); // edge0 buffer
            fProxyN b = arena.fProxyLinVec(n, (fProxy)2, (fProxy)4);    // edge1 buffer (> edge0)
            fProxyN a0 = a.Copy();
            fProxyN b0 = b.Copy();
            fProxy t = (fProxy)0.5;

            a.smoothstepInPlace(b, t); // a[i] = smoothstep(a[i], b[i], t)

            for (int i = 0; i < n; i++)
            {
                AssertClose(a[i], math.smoothstep(a0[i], b0[i], t));
                AssertClose(b[i], b0[i]);
            }
        }

        private void SmoothstepScalarEdgesTest(ref Arena arena)
        {
            int n = 24;
            fProxyN x = arena.fProxyLinVec(n, (fProxy)(-0.5), (fProxy)1.5); // spans below/above edges
            fProxyN x0 = x.Copy();
            fProxy edge0 = (fProxy)0;
            fProxy edge1 = (fProxy)1;

            x.smoothstepInPlace(edge0, edge1); // x[i] = smoothstep(edge0, edge1, x[i])

            for (int i = 0; i < n; i++)
                AssertClose(x[i], math.smoothstep(edge0, edge1, x0[i]));
        }

        private void StepTest(ref Arena arena)
        {
            int n = 24;
            fProxyN x = arena.fProxyLinVec(n, (fProxy)(-1), (fProxy)2);
            fProxyN x0 = x.Copy();
            fProxy edge = (fProxy)0.5;

            x.stepInPlace(edge); // x[i] = step(edge, x[i]) == (x >= edge ? 1 : 0)

            for (int i = 0; i < n; i++)
                AssertClose(x[i], math.step(edge, x0[i]));
        }

        private void MadTest(ref Arena arena)
        {
            int n = 24;
            fProxyN a = arena.fProxyLinVec(n, (fProxy)(-2), (fProxy)2);
            fProxyN b = arena.fProxyLinVec(n, (fProxy)1, (fProxy)3);
            fProxyN c = arena.fProxyLinVec(n, (fProxy)(-1), (fProxy)1);
            fProxyN a0 = a.Copy();
            fProxyN b0 = b.Copy();
            fProxyN c0 = c.Copy();

            a.madInPlace(b, c); // a[i] = a*b + c ; ONLY a is mutated

            for (int i = 0; i < n; i++)
            {
                AssertClose(a[i], a0[i] * b0[i] + c0[i]);
                AssertClose(b[i], b0[i]);
                AssertClose(c[i], c0[i]);
            }
        }

        private void RemapTest(ref Arena arena)
        {
            int n = 24;
            fProxyN x = arena.fProxyLinVec(n, (fProxy)1, (fProxy)9); // strictly inside [oldMin,oldMax]
            fProxyN x0 = x.Copy();
            fProxy oldMin = (fProxy)0, oldMax = (fProxy)10, newMin = (fProxy)(-1), newMax = (fProxy)1;

            x.remapInPlace(oldMin, oldMax, newMin, newMax);

            // CORRECT semantics per the wrapper's parameter names / doc: map x from [oldMin,oldMax]
            // onto [newMin,newMax]. Unity's math.remap takes the VALUE as its LAST argument:
            //   remap(srcStart, srcEnd, dstStart, dstEnd, value).
            // NOTE: this asserts the intended behaviour and currently FAILS - the kernel
            // (UnsafeMathOP.remap) calls math.remap(x, oldMin, oldMax, newMin, newMax), i.e. it
            // passes the value FIRST, so the arguments are rotated and the result is wrong. See the
            // agent report: this is a genuine argument-order bug in the production kernel, not a
            // test defect. The oracle here is the correct value the kernel should produce.
            for (int i = 0; i < n; i++)
                AssertClose(x[i], math.remap(oldMin, oldMax, newMin, newMax, x0[i]));
        }

        private void DegreesRadiansTest(ref Arena arena)
        {
            int n = 20;

            // degreesInPlace vs math.degrees
            fProxyN r = arena.fProxyLinVec(n, (fProxy)(-3), (fProxy)3);
            fProxyN r0 = r.Copy();
            fProxyN d = r.Copy();
            d.degreesInPlace();
            for (int i = 0; i < n; i++)
                AssertClose(d[i], math.degrees(r0[i]));

            // radians undoes degrees -> round trip back to original radians
            d.radiansInPlace();
            for (int i = 0; i < n; i++)
                AssertClose(d[i], r0[i]);

            // radiansInPlace vs math.radians on a degree-scale input
            fProxyN deg = arena.fProxyLinVec(n, (fProxy)(-180), (fProxy)180);
            fProxyN deg0 = deg.Copy();
            deg.radiansInPlace();
            for (int i = 0; i < n; i++)
                AssertClose(deg[i], math.radians(deg0[i]));
        }

        private void Atan2Test(ref Arena arena)
        {
            int n = 24;
            // receiver is y (numerator), argument is x (denominator): atan2InPlace(y, x) == atan2(y, x).
            fProxyN y = arena.fProxyLinVec(n, (fProxy)(-3), (fProxy)3);
            fProxyN x = arena.fProxyLinVec(n, (fProxy)0.5, (fProxy)4); // x > 0 (principal branch)
            fProxyN y0 = y.Copy();
            fProxyN x0 = x.Copy();

            y.atan2InPlace(x); // y[i] = atan2(y[i], x[i]) ; x untouched

            for (int i = 0; i < n; i++)
            {
                AssertClose(y[i], math.atan2(y0[i], x0[i]));
                AssertClose(x[i], x0[i]);
            }
        }

        private void MinBufTest(ref Arena arena)
        {
            int n = 24;
            fProxyN x = arena.fProxyLinVec(n, (fProxy)(-2), (fProxy)5);
            fProxyN y = arena.fProxyLinVec(n, (fProxy)4, (fProxy)(-1)); // crosses x so both branches hit
            fProxyN x0 = x.Copy();
            fProxyN y0 = y.Copy();

            x.minInPlace(y); // x[i] = min(x[i], y[i]) ; y untouched

            for (int i = 0; i < n; i++)
            {
                AssertClose(x[i], math.min(x0[i], y0[i]));
                AssertClose(y[i], y0[i]);
            }
        }

        private void MaxBufTest(ref Arena arena)
        {
            int n = 24;
            fProxyN x = arena.fProxyLinVec(n, (fProxy)(-2), (fProxy)5);
            fProxyN y = arena.fProxyLinVec(n, (fProxy)4, (fProxy)(-1));
            fProxyN x0 = x.Copy();
            fProxyN y0 = y.Copy();

            x.maxInPlace(y); // x[i] = max(x[i], y[i]) ; y untouched

            for (int i = 0; i < n; i++)
            {
                AssertClose(x[i], math.max(x0[i], y0[i]));
                AssertClose(y[i], y0[i]);
            }
        }

        // ---- absDiff/sqrDiff: componentwise |a-b| / (a-b)^2 (renamed from the kernel's old
        //      distance/distancesq names - these are NOT whole-vector Euclidean distances, each index
        //      is an independent scalar difference). Oracle computed directly, not via math.distance/
        //      math.distancesq, so the test doesn't just re-assert the kernel's own implementation. ----

        private void AbsDiffTest(ref Arena arena)
        {
            int n = 24;
            fProxyN x = arena.fProxyLinVec(n, (fProxy)(-2), (fProxy)5);
            fProxyN y = arena.fProxyLinVec(n, (fProxy)4, (fProxy)(-1)); // crosses x so sign of (x-y) varies
            fProxyN x0 = x.Copy();
            fProxyN y0 = y.Copy();

            x.absDiffInPlace(y); // x[i] = |x[i] - y[i]| ; y untouched

            for (int i = 0; i < n; i++)
            {
                AssertClose(x[i], math.abs(x0[i] - y0[i]));
                AssertClose(y[i], y0[i]);
            }
        }

        private void SqrDiffTest(ref Arena arena)
        {
            int n = 24;
            fProxyN x = arena.fProxyLinVec(n, (fProxy)(-2), (fProxy)5);
            fProxyN y = arena.fProxyLinVec(n, (fProxy)4, (fProxy)(-1));
            fProxyN x0 = x.Copy();
            fProxyN y0 = y.Copy();

            x.sqrDiffInPlace(y); // x[i] = (x[i] - y[i])^2 ; y untouched

            for (int i = 0; i < n; i++)
            {
                AssertClose(x[i], (x0[i] - y0[i]) * (x0[i] - y0[i]));
                AssertClose(y[i], y0[i]);
            }
        }

        private void SincosTest(ref Arena arena)
        {
            int n = 30;
            fProxyN x = arena.fProxyLinVec(n, (fProxy)(-3), (fProxy)3);
            fProxyN x0 = x.Copy();
            fProxyN s = arena.fProxyVec(n);
            fProxyN c = arena.fProxyVec(n);

            x.sincos(s, c); // x UNCHANGED; s <- sin(x), c <- cos(x). Renamed from sincosInPlace -
                            // review flagged the InPlace suffix as misleading since x isn't mutated.

            for (int i = 0; i < n; i++)
            {
                AssertClose(x[i], x0[i]); // the one op here that does NOT mutate its receiver
                AssertClose(s[i], math.sin(x0[i]));
                AssertClose(c[i], math.cos(x0[i]));
            }
        }

        // ---- matrix path: proves the generic IUnsafefProxyArray constraint covers fProxyMxN.
        //      Non-square (5x7) so the flat length, not a square dim, drives the loop. ----

        private void MatrixAbsTest(ref Arena arena)
        {
            int rows = 5, cols = 7;
            fProxyMxN m = arena.fProxyMat(rows, cols);
            for (int i = 0; i < m.Length; i++)
                m[i] = (fProxy)(i - 15) * (fProxy)0.5; // negatives and positives
            fProxyMxN m0 = m.Copy();

            m.absInPlace();

            for (int i = 0; i < m.Length; i++)
                AssertClose(m[i], math.abs(m0[i]));
        }

        private void MatrixExpTest(ref Arena arena)
        {
            int rows = 5, cols = 7;
            fProxyMxN m = arena.fProxyMat(rows, cols);
            for (int i = 0; i < m.Length; i++)
                m[i] = (fProxy)(i - 15) * (fProxy)0.1;
            fProxyMxN m0 = m.Copy();

            m.expInPlace();

            for (int i = 0; i < m.Length; i++)
                AssertClose(m[i], math.exp(m0[i]));
        }

        private void MatrixMadTest(ref Arena arena)
        {
            int rows = 5, cols = 7;
            fProxyMxN a = arena.fProxyMat(rows, cols);
            fProxyMxN b = arena.fProxyMat(rows, cols);
            fProxyMxN c = arena.fProxyMat(rows, cols);
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = (fProxy)(i - 15) * (fProxy)0.2;
                b[i] = (fProxy)(i) * (fProxy)0.3 - (fProxy)1;
                c[i] = (fProxy)2 - (fProxy)i * (fProxy)0.1;
            }
            fProxyMxN a0 = a.Copy();
            fProxyMxN b0 = b.Copy();
            fProxyMxN c0 = c.Copy();

            a.madInPlace(b, c); // a = a*b + c ; only a mutated

            for (int i = 0; i < a.Length; i++)
            {
                AssertClose(a[i], a0[i] * b0[i] + c0[i]);
                AssertClose(b[i], b0[i]);
                AssertClose(c[i], c0[i]);
            }
        }

        // ---- degenerate shapes ----

        private void SingleElementTest(ref Arena arena)
        {
            fProxyN v = arena.fProxyVec(1, (fProxy)(-2.5));
            v.absInPlace();
            AssertClose(v[0], (fProxy)2.5);

            fProxyN a = arena.fProxyVec(1, (fProxy)3);
            fProxyN b = arena.fProxyVec(1, (fProxy)4);
            fProxyN c = arena.fProxyVec(1, (fProxy)5);
            a.madInPlace(b, c);
            AssertClose(a[0], (fProxy)17); // 3*4 + 5

            fProxyN x = arena.fProxyVec(1, (fProxy)0.7);
            fProxyN s = arena.fProxyVec(1);
            fProxyN co = arena.fProxyVec(1);
            x.sincos(s, co);
            AssertClose(x[0], (fProxy)0.7);
            AssertClose(s[0], math.sin((fProxy)0.7));
            AssertClose(co[0], math.cos((fProxy)0.7));
        }

        private void EmptyBufferTest(ref Arena arena)
        {
            // zero-length buffers: kernels loop zero times, must not touch memory / throw.
            fProxyN v = arena.fProxyVec(0);
            fProxyN y = arena.fProxyVec(0);
            v.absInPlace();
            v.sinInPlace();
            v.reluInPlace();
            v.minInPlace(y);
            v.powInPlace(2);
            Assert.AreEqual(0, v.N);
        }
    }

    public static Array GetEnums()
    {
        return Enum.GetValues(typeof(TestsJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void Test(TestsJob.TestType type)
    {
        new TestsJob() { Type = type }.Run();
    }
}
