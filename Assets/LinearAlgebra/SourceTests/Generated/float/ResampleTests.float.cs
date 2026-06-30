using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for floatResample_OP (data-vector interpolation + 1D/2D resizing).
//
// Verification mixes EXACT checks (integer-position sampling, endpoint pinning, aligned-grid identity,
// documented edge-mode taps via Nearest) with property checks at a per-precision tolerance that scales
// with Consts.floatSqrtEps (loose for float, tight for double) for the polynomial-reproduction and
// planar-field cases.
//
// Catmull-Rom reproduces cubic polynomials exactly on a uniform grid; bilinear (two linear passes)
// reproduces an affine/planar field exactly — those are the structural identities exercised here.
//
// In-job (Burst) tests cover values; managed-thread Assert.Throws tests cover the validation paths.
public class floatResampleTests
{
    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            IntegerPositionsAllInterps,
            LinearMidpointMean,
            CubicReproducesPolynomial,
            EdgeModeClampTaps,
            EdgeModeWrapTaps,
            EdgeModeMirrorTaps,
            SampleAtIntoGather,
            ResampleIdentityAlignedGrid,
            ResampleEndpointsPreserved,
            ResampleLinearRampStaysLinear,
            ResampleSingleDest,
            Resample2DIdentity,
            Resample2DCornersPreserved,
            Resample2DBilinearPlanar,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<float> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.IntegerPositionsAllInterps:   IntegerPositionsAllInterps();   break;
                case TestType.LinearMidpointMean:           LinearMidpointMean();           break;
                case TestType.CubicReproducesPolynomial:    CubicReproducesPolynomial();    break;
                case TestType.EdgeModeClampTaps:            EdgeModeClampTaps();            break;
                case TestType.EdgeModeWrapTaps:             EdgeModeWrapTaps();            break;
                case TestType.EdgeModeMirrorTaps:           EdgeModeMirrorTaps();          break;
                case TestType.SampleAtIntoGather:           SampleAtIntoGather();           break;
                case TestType.ResampleIdentityAlignedGrid:  ResampleIdentityAlignedGrid();  break;
                case TestType.ResampleEndpointsPreserved:   ResampleEndpointsPreserved();   break;
                case TestType.ResampleLinearRampStaysLinear:ResampleLinearRampStaysLinear();break;
                case TestType.ResampleSingleDest:           ResampleSingleDest();           break;
                case TestType.Resample2DIdentity:           Resample2DIdentity();           break;
                case TestType.Resample2DCornersPreserved:   Resample2DCornersPreserved();   break;
                case TestType.Resample2DBilinearPlanar:     Resample2DBilinearPlanar();     break;
            }
        }

        // =====================================================================
        // sampleAt
        // =====================================================================

        // At INTEGER positions 0, 1, N-1 every interp returns the exact data value (Nearest rounds to
        // self; Linear has frac==0; Cubic has t==0 -> 0.5*2*p1 == p1). Bit-exact.
        void IntegerPositionsAllInterps()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var data = arena.floatVec(n);
            data[0] = (float)10; data[1] = (float)20; data[2] = (float)30;
            data[3] = (float)40; data[4] = (float)50;

            // probe positions 0, 1, and N-1
            CheckIntegerPos(in data, 0);
            CheckIntegerPos(in data, 1);
            CheckIntegerPos(in data, n - 1);

            arena.Dispose();
        }

        void CheckIntegerPos(in floatN data, int ix)
        {
            float pos = (float)ix;
            AssertClose(floatResample_OP.sampleAt(in data, pos, Interp.Nearest, EdgeMode.Clamp), data[ix], (float)0);
            AssertClose(floatResample_OP.sampleAt(in data, pos, Interp.Linear,  EdgeMode.Clamp), data[ix], (float)0);
            AssertClose(floatResample_OP.sampleAt(in data, pos, Interp.Cubic,   EdgeMode.Clamp), data[ix], (float)0);
        }

        // Linear at pos=0.5 -> exact mean of the two neighbors.
        void LinearMidpointMean()
        {
            var arena = new Arena(Allocator.Persistent);

            var data = arena.floatVec(4);
            data[0] = (float)10; data[1] = (float)20; data[2] = (float)33; data[3] = (float)40;

            float mid01 = floatResample_OP.sampleAt(in data, (float)0.5, Interp.Linear, EdgeMode.Clamp);
            AssertClose(mid01, (float)15, (float)10 * Consts.floatSqrtEps);   // (10+20)/2

            float mid12 = floatResample_OP.sampleAt(in data, (float)1.5, Interp.Linear, EdgeMode.Clamp);
            AssertClose(mid12, (float)26.5, (float)10 * Consts.floatSqrtEps); // (20+33)/2

            arena.Dispose();
        }

        // Catmull-Rom (Keys cubic convolution, a=-0.5) is third-order accurate: it reproduces
        // polynomials up to degree 2 EXACTLY on a uniform grid, but NOT a general cubic (the
        // centered-difference tangent (f(i+1)-f(i-1))/2 = f'(i) + f'''(i)/6 is inexact when f'''!=0).
        // So the precision property to assert is exact QUADRATIC reproduction. Sample at interior
        // non-integer positions (all 4 taps in-range, so the edge mode is irrelevant) and compare to f.
        void CubicReproducesPolynomial()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var data = arena.floatVec(n);
            for (int i = 0; i < n; i++) data[i] = Quad((float)i);

            float tol = (float)100 * Consts.floatSqrtEps;   // float ~3.5e-2, double ~1.5e-6
            CheckQuadAt(in data, (float)1.3, tol);
            CheckQuadAt(in data, (float)2.5, tol);
            CheckQuadAt(in data, (float)3.4, tol);

            arena.Dispose();
        }

        void CheckQuadAt(in floatN data, float pos, float tol)
        {
            float got = floatResample_OP.sampleAt(in data, pos, Interp.Cubic, EdgeMode.Clamp);
            AssertClose(got, Quad(pos), tol);
        }

        float Quad(float x) =>
            (float)0.3 * x * x - (float)1.2 * x + (float)2.0;

        // Documented edge taps via Nearest at integer out-of-range positions (round(pos)==pos), reading
        // back distinct data values. data[i] = i+1 over N=5.
        void EdgeModeClampTaps()
        {
            var arena = new Arena(Allocator.Persistent);
            var data = Ramp(ref arena, 5);   // {1,2,3,4,5}

            // Clamp repeats the edge: idx(-1)=0, idx(7)=4.
            AssertClose(Near(in data, -1, EdgeMode.Clamp), (float)1, (float)0);
            AssertClose(Near(in data, -3, EdgeMode.Clamp), (float)1, (float)0);
            AssertClose(Near(in data,  7, EdgeMode.Clamp), (float)5, (float)0);

            arena.Dispose();
        }

        void EdgeModeWrapTaps()
        {
            var arena = new Arena(Allocator.Persistent);
            var data = Ramp(ref arena, 5);

            // Wrap is periodic: idx(-1)=4, idx(5)=0, idx(6)=1.
            AssertClose(Near(in data, -1, EdgeMode.Wrap), (float)5, (float)0);
            AssertClose(Near(in data,  5, EdgeMode.Wrap), (float)1, (float)0);
            AssertClose(Near(in data,  6, EdgeMode.Wrap), (float)2, (float)0);

            arena.Dispose();
        }

        void EdgeModeMirrorTaps()
        {
            var arena = new Arena(Allocator.Persistent);
            var data = Ramp(ref arena, 5);

            // Mirror reflect101 (N=5): idx(-1)=1, idx(5)=3, idx(8)=0, idx(-4)=4.
            AssertClose(Near(in data, -1, EdgeMode.Mirror), (float)2, (float)0);
            AssertClose(Near(in data,  5, EdgeMode.Mirror), (float)4, (float)0);
            AssertClose(Near(in data,  8, EdgeMode.Mirror), (float)1, (float)0);
            AssertClose(Near(in data, -4, EdgeMode.Mirror), (float)5, (float)0);

            arena.Dispose();
        }

        floatN Ramp(ref Arena arena, int n)
        {
            var v = arena.floatVec(n);
            for (int i = 0; i < n; i++) v[i] = (float)(i + 1);
            return v;
        }

        // Nearest sample at an integer position (so round(pos)==pos) — reads back data[idx(pos)].
        float Near(in floatN data, int pos, EdgeMode edge) =>
            floatResample_OP.sampleAt(in data, (float)pos, Interp.Nearest, edge);

        // =====================================================================
        // sampleAtInto
        // =====================================================================

        // Gather matches per-position sampleAt, element for element.
        void SampleAtIntoGather()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 6;
            var data = arena.floatVec(n);
            for (int i = 0; i < n; i++) data[i] = (float)(i * i);   // 0,1,4,9,16,25

            int k = 5;
            var positions = arena.floatVec(k);
            positions[0] = (float)0;
            positions[1] = (float)1.5;
            positions[2] = (float)2.25;
            positions[3] = (float)4.0;
            positions[4] = (float)(-1);   // exercises the edge mode

            var dest = arena.floatVec(k);
            floatResample_OP.sampleAtInto(in data, in positions, ref dest, Interp.Cubic, EdgeMode.Mirror);

            for (int j = 0; j < k; j++)
            {
                float expected = floatResample_OP.sampleAt(in data, positions[j], Interp.Cubic, EdgeMode.Mirror);
                AssertClose(dest[j], expected, (float)0);   // identical code path -> bit-exact
            }

            arena.Dispose();
        }

        // =====================================================================
        // resampleInto
        // =====================================================================

        // dst.N == src.N -> pos(j) == j (integer), so Linear AND Cubic reproduce the input exactly.
        void ResampleIdentityAlignedGrid()
        {
            var arena = new Arena(Allocator.Persistent);

            int n = 5;
            var src = arena.floatVec(n);
            src[0] = (float)2.5; src[1] = (float)(-1.0); src[2] = (float)3.7;
            src[3] = (float)0.2; src[4] = (float)5.5;

            var dstL = arena.floatVec(n);
            floatResample_OP.resampleInto(in src, ref dstL, Interp.Linear, EdgeMode.Clamp);
            for (int i = 0; i < n; i++) AssertClose(dstL[i], src[i], (float)0);

            var dstC = arena.floatVec(n);
            floatResample_OP.resampleInto(in src, ref dstC, Interp.Cubic, EdgeMode.Clamp);
            for (int i = 0; i < n; i++) AssertClose(dstC[i], src[i], (float)0);

            arena.Dispose();
        }

        // Endpoints pinned bit-exact on both up- and down-sample: dst[0]==src[0], dst[last]==src[last].
        void ResampleEndpointsPreserved()
        {
            var arena = new Arena(Allocator.Persistent);

            // upsample 4 -> 9
            var up = arena.floatVec(4);
            up[0] = (float)10; up[1] = (float)20; up[2] = (float)30; up[3] = (float)40;
            var dstUp = arena.floatVec(9);
            floatResample_OP.resampleInto(in up, ref dstUp, Interp.Cubic, EdgeMode.Clamp);
            AssertClose(dstUp[0], up[0], (float)0);
            AssertClose(dstUp[8], up[3], (float)0);

            // downsample 9 -> 4
            var down = arena.floatVec(9);
            for (int i = 0; i < 9; i++) down[i] = (float)(i * i);
            var dstDown = arena.floatVec(4);
            floatResample_OP.resampleInto(in down, ref dstDown, Interp.Linear, EdgeMode.Clamp);
            AssertClose(dstDown[0], down[0], (float)0);
            AssertClose(dstDown[3], down[8], (float)0);

            arena.Dispose();
        }

        // Linear upsample of a linear ramp stays linear: dst[j] == a*pos(j)+b, pos(j)=j*(srcN-1)/(dstN-1).
        void ResampleLinearRampStaysLinear()
        {
            var arena = new Arena(Allocator.Persistent);

            float a = (float)2, b = (float)(-1);
            int srcN = 4, dstN = 10;
            var src = arena.floatVec(srcN);
            for (int i = 0; i < srcN; i++) src[i] = a * (float)i + b;   // -1,1,3,5

            var dst = arena.floatVec(dstN);
            floatResample_OP.resampleInto(in src, ref dst, Interp.Linear, EdgeMode.Clamp);

            float scale = (float)(srcN - 1) / (float)(dstN - 1);
            float tol = (float)20 * Consts.floatSqrtEps;
            for (int j = 0; j < dstN; j++)
            {
                float pos = (float)j * scale;
                AssertClose(dst[j], a * pos + b, tol);
            }

            arena.Dispose();
        }

        // dst.N == 1 -> dst[0] == src[0].
        void ResampleSingleDest()
        {
            var arena = new Arena(Allocator.Persistent);

            var src = arena.floatVec(3);
            src[0] = (float)5; src[1] = (float)6; src[2] = (float)7;

            var dst = arena.floatVec(1);
            floatResample_OP.resampleInto(in src, ref dst, Interp.Linear, EdgeMode.Clamp);
            AssertClose(dst[0], src[0], (float)0);

            arena.Dispose();
        }

        // =====================================================================
        // resample2DInto
        // =====================================================================

        // M x N -> M x N: aligned grid, so Nearest/Linear/Cubic all reproduce the source exactly.
        void Resample2DIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 3, n = 4;
            var src = arena.floatMat(m, n);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    src[r, c] = (float)(r * 10 + c) - (float)3.5;

            CheckIdentity2D(ref arena, in src, Interp.Nearest);
            CheckIdentity2D(ref arena, in src, Interp.Linear);
            CheckIdentity2D(ref arena, in src, Interp.Cubic);

            arena.Dispose();
        }

        void CheckIdentity2D(ref Arena arena, in floatMxN src, Interp interp)
        {
            var dst = arena.floatMat(src.M_Rows, src.N_Cols);
            floatResample_OP.resample2DInto(in src, ref dst, interp, EdgeMode.Clamp);
            for (int r = 0; r < src.M_Rows; r++)
                for (int c = 0; c < src.N_Cols; c++)
                    AssertClose(dst[r, c], src[r, c], (float)0);
        }

        // The 4 corners are pinned exactly on an arbitrary resize.
        void Resample2DCornersPreserved()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 3, n = 4;
            var src = arena.floatMat(m, n);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    src[r, c] = (float)(r * 7 - c * 3) + (float)0.25;

            int M2 = 5, N2 = 7;
            var dst = arena.floatMat(M2, N2);
            floatResample_OP.resample2DInto(in src, ref dst, Interp.Cubic, EdgeMode.Clamp);

            AssertClose(dst[0, 0],        src[0, 0],         (float)0);
            AssertClose(dst[0, N2 - 1],   src[0, n - 1],     (float)0);
            AssertClose(dst[M2 - 1, 0],   src[m - 1, 0],     (float)0);
            AssertClose(dst[M2 - 1, N2-1],src[m - 1, n - 1], (float)0);

            arena.Dispose();
        }

        // Bilinear (Linear, two separable passes) of a planar field f(r,c)=a*r+b*c+d is exact within
        // a per-precision tolerance. src 4x5 -> dst 7x9: rowPos(i)=i*0.5, colPos(j)=j*0.5.
        void Resample2DBilinearPlanar()
        {
            var arena = new Arena(Allocator.Persistent);

            float a = (float)1.5, b = (float)(-0.7), d = (float)2;
            int m = 4, n = 5;
            var src = arena.floatMat(m, n);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    src[r, c] = a * (float)r + b * (float)c + d;

            int M2 = 7, N2 = 9;
            var dst = arena.floatMat(M2, N2);
            floatResample_OP.resample2DInto(in src, ref dst, Interp.Linear, EdgeMode.Clamp);

            float rScale = (float)(m - 1) / (float)(M2 - 1);
            float cScale = (float)(n - 1) / (float)(N2 - 1);
            float tol = (float)50 * Consts.floatSqrtEps;
            for (int i = 0; i < M2; i++)
                for (int j = 0; j < N2; j++)
                {
                    float rp = (float)i * rScale;
                    float cp = (float)j * cScale;
                    AssertClose(dst[i, j], a * rp + b * cp + d, tol);
                }

            arena.Dispose();
        }

        // =====================================================================
        // helpers (Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff)
        // =====================================================================

        void AssertClose(float a, float b, float precision)
        {
            float diff = math.abs(a - b);
            if (!(diff <= precision) && Fail[0] == (float)0)
            {
                Fail[0] = (float)1;
                Fail[1] = a;
                Fail[2] = b;
                Fail[3] = diff;
            }
            Assert.IsTrue(diff <= precision);
        }
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void ResampleTests(TestJob.TestType type)
    {
        var fail = new NativeArray<float>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e)
        {
            if (fail[0] != (float)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally
        {
            fail.Dispose();
        }
    }

    // ---------------- Managed validation throws (main thread, not in a Burst job) ----------------

    [Test]
    public void SampleAtEmptyThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var empty = arena.floatVec(0);
            Assert.Throws<ArgumentException>(
                () => floatResample_OP.sampleAt(in empty, (float)0, Interp.Linear, EdgeMode.Clamp));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void SampleAtIntoValidates()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var data = arena.floatVec(5);
            for (int i = 0; i < 5; i++) data[i] = (float)i;

            // dest.N != positions.N
            var positions = arena.floatVec(3);
            var destBad = arena.floatVec(4);
            Assert.Throws<ArgumentException>(
                () => floatResample_OP.sampleAtInto(in data, in positions, ref destBad, Interp.Linear, EdgeMode.Clamp));

            // empty data (dest.N == positions.N so it reaches the data check) -> "sampleAtInto:" message
            var emptyData = arena.floatVec(0);
            var dest = arena.floatVec(3);
            var ex = Assert.Throws<ArgumentException>(
                () => floatResample_OP.sampleAtInto(in emptyData, in positions, ref dest, Interp.Linear, EdgeMode.Clamp));
            StringAssert.Contains("sampleAtInto:", ex.Message);
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void ResampleIntoValidates()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var src = arena.floatVec(4);
            var emptyDst = arena.floatVec(0);
            Assert.Throws<ArgumentException>(
                () => floatResample_OP.resampleInto(in src, ref emptyDst, Interp.Linear, EdgeMode.Clamp));

            var emptySrc = arena.floatVec(0);
            var dst = arena.floatVec(4);
            Assert.Throws<ArgumentException>(
                () => floatResample_OP.resampleInto(in emptySrc, ref dst, Interp.Linear, EdgeMode.Clamp));
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void Resample2DIntoValidates()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var src = arena.floatMat(3, 3);

            // dst with 0 rows
            var dstNoRows = arena.floatMat(0, 3);
            Assert.Throws<ArgumentException>(
                () => floatResample_OP.resample2DInto(in src, ref dstNoRows, Interp.Linear, EdgeMode.Clamp));

            // dst with 0 cols
            var dstNoCols = arena.floatMat(3, 0);
            Assert.Throws<ArgumentException>(
                () => floatResample_OP.resample2DInto(in src, ref dstNoCols, Interp.Linear, EdgeMode.Clamp));

            // src 0x0 (validated before any scratch allocation)
            var emptySrc = arena.floatMat(0, 0);
            var dst = arena.floatMat(2, 2);
            Assert.Throws<ArgumentException>(
                () => floatResample_OP.resample2DInto(in emptySrc, ref dst, Interp.Linear, EdgeMode.Clamp));
        }
        finally { arena.Dispose(); }
    }
}
