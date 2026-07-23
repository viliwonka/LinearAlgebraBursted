using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Tests for Resample (data-vector interpolation + 1D/2D resizing).
//
// Verification mixes EXACT checks (integer-position sampling, endpoint pinning, aligned-grid identity,
// documented edge-mode taps via Nearest) with property checks at a per-precision tolerance that scales
// with Consts.fProxySqrtEps (loose for float, tight for double) for the polynomial-reproduction and
// planar-field cases.
//
// Catmull-Rom reproduces cubic polynomials exactly on a uniform grid; bilinear (two linear passes)
// reproduces an affine/planar field exactly — those are the structural identities exercised here.
//
// In-job (Burst) tests cover values; managed-thread Assert.Throws tests cover the validation paths.
public class fProxyResampleTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
        public NativeArray<fProxy> Fail;

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
            int n = 5;
            var data = new fProxyN(n, Allocator.Temp);
            data[0] = (fProxy)10; data[1] = (fProxy)20; data[2] = (fProxy)30;
            data[3] = (fProxy)40; data[4] = (fProxy)50;

            CheckIntegerPos(in data, 0);
            CheckIntegerPos(in data, 1);
            CheckIntegerPos(in data, n - 1);
        }

        void CheckIntegerPos(in fProxyN data, int ix)
        {
            fProxy pos = (fProxy)ix;
            AssertClose(Resample.sampleAt(in data, pos, Interp.Nearest, EdgeMode.Clamp), data[ix], (fProxy)0);
            AssertClose(Resample.sampleAt(in data, pos, Interp.Linear,  EdgeMode.Clamp), data[ix], (fProxy)0);
            AssertClose(Resample.sampleAt(in data, pos, Interp.Cubic,   EdgeMode.Clamp), data[ix], (fProxy)0);
        }

        // Linear at pos=0.5 -> exact mean of the two neighbors.
        void LinearMidpointMean()
        {
            var data = new fProxyN(4, Allocator.Temp);
            data[0] = (fProxy)10; data[1] = (fProxy)20; data[2] = (fProxy)33; data[3] = (fProxy)40;

            fProxy mid01 = Resample.sampleAt(in data, (fProxy)0.5, Interp.Linear, EdgeMode.Clamp);
            AssertClose(mid01, (fProxy)15, (fProxy)10 * Consts.fProxySqrtEps);   // (10+20)/2

            fProxy mid12 = Resample.sampleAt(in data, (fProxy)1.5, Interp.Linear, EdgeMode.Clamp);
            AssertClose(mid12, (fProxy)26.5, (fProxy)10 * Consts.fProxySqrtEps); // (20+33)/2
        }

        // Catmull-Rom (Keys cubic convolution, a=-0.5) is third-order accurate: it reproduces
        // polynomials up to degree 2 EXACTLY on a uniform grid, but NOT a general cubic (the
        // centered-difference tangent (f(i+1)-f(i-1))/2 = f'(i) + f'''(i)/6 is inexact when f'''!=0).
        // So the precision property to assert is exact QUADRATIC reproduction. Sample at interior
        // non-integer positions (all 4 taps in-range, so the edge mode is irrelevant) and compare to f.
        void CubicReproducesPolynomial()
        {
            int n = 6;
            var data = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) data[i] = Quad((fProxy)i);

            fProxy tol = (fProxy)100 * Consts.fProxySqrtEps;   // float ~3.5e-2, double ~1.5e-6
            CheckQuadAt(in data, (fProxy)1.3, tol);
            CheckQuadAt(in data, (fProxy)2.5, tol);
            CheckQuadAt(in data, (fProxy)3.4, tol);
        }

        void CheckQuadAt(in fProxyN data, fProxy pos, fProxy tol)
        {
            fProxy got = Resample.sampleAt(in data, pos, Interp.Cubic, EdgeMode.Clamp);
            AssertClose(got, Quad(pos), tol);
        }

        fProxy Quad(fProxy x) =>
            (fProxy)0.3 * x * x - (fProxy)1.2 * x + (fProxy)2.0;

        // Documented edge taps via Nearest at integer out-of-range positions (round(pos)==pos), reading
        // back distinct data values. data[i] = i+1 over N=5.
        void EdgeModeClampTaps()
        {
            var data = Ramp(5);   // {1,2,3,4,5}

            // Clamp repeats the edge: idx(-1)=0, idx(7)=4.
            AssertClose(Near(in data, -1, EdgeMode.Clamp), (fProxy)1, (fProxy)0);
            AssertClose(Near(in data, -3, EdgeMode.Clamp), (fProxy)1, (fProxy)0);
            AssertClose(Near(in data,  7, EdgeMode.Clamp), (fProxy)5, (fProxy)0);
        }

        void EdgeModeWrapTaps()
        {
            var data = Ramp(5);

            // Wrap is periodic: idx(-1)=4, idx(5)=0, idx(6)=1.
            AssertClose(Near(in data, -1, EdgeMode.Wrap), (fProxy)5, (fProxy)0);
            AssertClose(Near(in data,  5, EdgeMode.Wrap), (fProxy)1, (fProxy)0);
            AssertClose(Near(in data,  6, EdgeMode.Wrap), (fProxy)2, (fProxy)0);
        }

        void EdgeModeMirrorTaps()
        {
            var data = Ramp(5);

            // Mirror reflect101 (N=5): idx(-1)=1, idx(5)=3, idx(8)=0, idx(-4)=4.
            AssertClose(Near(in data, -1, EdgeMode.Mirror), (fProxy)2, (fProxy)0);
            AssertClose(Near(in data,  5, EdgeMode.Mirror), (fProxy)4, (fProxy)0);
            AssertClose(Near(in data,  8, EdgeMode.Mirror), (fProxy)1, (fProxy)0);
            AssertClose(Near(in data, -4, EdgeMode.Mirror), (fProxy)5, (fProxy)0);
        }

        fProxyN Ramp(int n)
        {
            var v = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) v[i] = (fProxy)(i + 1);
            return v;
        }

        // Nearest sample at an integer position (so round(pos)==pos) — reads back data[idx(pos)].
        fProxy Near(in fProxyN data, int pos, EdgeMode edge) =>
            Resample.sampleAt(in data, (fProxy)pos, Interp.Nearest, edge);

        // =====================================================================
        // sampleAtInto
        // =====================================================================

        // Gather matches per-position sampleAt, element for element.
        void SampleAtIntoGather()
        {
            int n = 6;
            var data = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) data[i] = (fProxy)(i * i);   // 0,1,4,9,16,25

            int k = 5;
            var positions = new fProxyN(k, Allocator.Temp);
            positions[0] = (fProxy)0;
            positions[1] = (fProxy)1.5;
            positions[2] = (fProxy)2.25;
            positions[3] = (fProxy)4.0;
            positions[4] = (fProxy)(-1);   // exercises the edge mode

            var dest = new fProxyN(k, Allocator.Temp);
            Resample.sampleAtInto(in data, in positions, ref dest, Interp.Cubic, EdgeMode.Mirror);

            for (int j = 0; j < k; j++)
            {
                fProxy expected = Resample.sampleAt(in data, positions[j], Interp.Cubic, EdgeMode.Mirror);
                AssertClose(dest[j], expected, (fProxy)0);   // identical code path -> bit-exact
            }
        }

        // =====================================================================
        // resampleInto
        // =====================================================================

        // dst.N == src.N -> pos(j) == j (integer), so Linear AND Cubic reproduce the input exactly.
        void ResampleIdentityAlignedGrid()
        {
            int n = 5;
            var src = new fProxyN(n, Allocator.Temp);
            src[0] = (fProxy)2.5; src[1] = (fProxy)(-1.0); src[2] = (fProxy)3.7;
            src[3] = (fProxy)0.2; src[4] = (fProxy)5.5;

            var dstL = new fProxyN(n, Allocator.Temp);
            Resample.resampleInto(in src, ref dstL, Interp.Linear, EdgeMode.Clamp);
            for (int i = 0; i < n; i++) AssertClose(dstL[i], src[i], (fProxy)0);

            var dstC = new fProxyN(n, Allocator.Temp);
            Resample.resampleInto(in src, ref dstC, Interp.Cubic, EdgeMode.Clamp);
            for (int i = 0; i < n; i++) AssertClose(dstC[i], src[i], (fProxy)0);
        }

        // Endpoints pinned bit-exact on both up- and down-sample: dst[0]==src[0], dst[last]==src[last].
        void ResampleEndpointsPreserved()
        {
            // upsample 4 -> 9
            var up = new fProxyN(4, Allocator.Temp);
            up[0] = (fProxy)10; up[1] = (fProxy)20; up[2] = (fProxy)30; up[3] = (fProxy)40;
            var dstUp = new fProxyN(9, Allocator.Temp);
            Resample.resampleInto(in up, ref dstUp, Interp.Cubic, EdgeMode.Clamp);
            AssertClose(dstUp[0], up[0], (fProxy)0);
            AssertClose(dstUp[8], up[3], (fProxy)0);

            // downsample 9 -> 4
            var down = new fProxyN(9, Allocator.Temp);
            for (int i = 0; i < 9; i++) down[i] = (fProxy)(i * i);
            var dstDown = new fProxyN(4, Allocator.Temp);
            Resample.resampleInto(in down, ref dstDown, Interp.Linear, EdgeMode.Clamp);
            AssertClose(dstDown[0], down[0], (fProxy)0);
            AssertClose(dstDown[3], down[8], (fProxy)0);
        }

        // Linear upsample of a linear ramp stays linear: dst[j] == a*pos(j)+b, pos(j)=j*(srcN-1)/(dstN-1).
        void ResampleLinearRampStaysLinear()
        {
            fProxy a = (fProxy)2, b = (fProxy)(-1);
            int srcN = 4, dstN = 10;
            var src = new fProxyN(srcN, Allocator.Temp);
            for (int i = 0; i < srcN; i++) src[i] = a * (fProxy)i + b;   // -1,1,3,5

            var dst = new fProxyN(dstN, Allocator.Temp);
            Resample.resampleInto(in src, ref dst, Interp.Linear, EdgeMode.Clamp);

            fProxy scale = (fProxy)(srcN - 1) / (fProxy)(dstN - 1);
            fProxy tol = (fProxy)20 * Consts.fProxySqrtEps;
            for (int j = 0; j < dstN; j++)
            {
                fProxy pos = (fProxy)j * scale;
                AssertClose(dst[j], a * pos + b, tol);
            }
        }

        // dst.N == 1 -> dst[0] == src[0].
        void ResampleSingleDest()
        {
            var src = new fProxyN(3, Allocator.Temp);
            src[0] = (fProxy)5; src[1] = (fProxy)6; src[2] = (fProxy)7;

            var dst = new fProxyN(1, Allocator.Temp);
            Resample.resampleInto(in src, ref dst, Interp.Linear, EdgeMode.Clamp);
            AssertClose(dst[0], src[0], (fProxy)0);
        }

        // =====================================================================
        // resample2DInto
        // =====================================================================

        // M x N -> M x N: aligned grid, so Nearest/Linear/Cubic all reproduce the source exactly.
        void Resample2DIdentity()
        {
            int m = 3, n = 4;
            var src = new fProxyMxN(m, n, Allocator.Temp);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    src[r, c] = (fProxy)(r * 10 + c) - (fProxy)3.5;

            CheckIdentity2D(in src, Interp.Nearest);
            CheckIdentity2D(in src, Interp.Linear);
            CheckIdentity2D(in src, Interp.Cubic);
        }

        void CheckIdentity2D(in fProxyMxN src, Interp interp)
        {
            var dst = new fProxyMxN(src.M_Rows, src.N_Cols, Allocator.Temp);
            Resample.resample2DInto(in src, ref dst, interp, EdgeMode.Clamp);
            for (int r = 0; r < src.M_Rows; r++)
                for (int c = 0; c < src.N_Cols; c++)
                    AssertClose(dst[r, c], src[r, c], (fProxy)0);
        }

        // The 4 corners are pinned exactly on an arbitrary resize.
        void Resample2DCornersPreserved()
        {
            int m = 3, n = 4;
            var src = new fProxyMxN(m, n, Allocator.Temp);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    src[r, c] = (fProxy)(r * 7 - c * 3) + (fProxy)0.25;

            int M2 = 5, N2 = 7;
            var dst = new fProxyMxN(M2, N2, Allocator.Temp);
            Resample.resample2DInto(in src, ref dst, Interp.Cubic, EdgeMode.Clamp);

            AssertClose(dst[0, 0],        src[0, 0],         (fProxy)0);
            AssertClose(dst[0, N2 - 1],   src[0, n - 1],     (fProxy)0);
            AssertClose(dst[M2 - 1, 0],   src[m - 1, 0],     (fProxy)0);
            AssertClose(dst[M2 - 1, N2-1],src[m - 1, n - 1], (fProxy)0);
        }

        // Bilinear (Linear, two separable passes) of a planar field f(r,c)=a*r+b*c+d is exact within
        // a per-precision tolerance. src 4x5 -> dst 7x9: rowPos(i)=i*0.5, colPos(j)=j*0.5.
        void Resample2DBilinearPlanar()
        {
            fProxy a = (fProxy)1.5, b = (fProxy)(-0.7), d = (fProxy)2;
            int m = 4, n = 5;
            var src = new fProxyMxN(m, n, Allocator.Temp);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    src[r, c] = a * (fProxy)r + b * (fProxy)c + d;

            int M2 = 7, N2 = 9;
            var dst = new fProxyMxN(M2, N2, Allocator.Temp);
            Resample.resample2DInto(in src, ref dst, Interp.Linear, EdgeMode.Clamp);

            fProxy rScale = (fProxy)(m - 1) / (fProxy)(M2 - 1);
            fProxy cScale = (fProxy)(n - 1) / (fProxy)(N2 - 1);
            fProxy tol = (fProxy)50 * Consts.fProxySqrtEps;
            for (int i = 0; i < M2; i++)
                for (int j = 0; j < N2; j++)
                {
                    fProxy rp = (fProxy)i * rScale;
                    fProxy cp = (fProxy)j * cScale;
                    AssertClose(dst[i, j], a * rp + b * cp + d, tol);
                }
        }

        // =====================================================================
        // helpers (Fail layout: [0]=flag, [1]=got, [2]=expected/limit, [3]=diff)
        // =====================================================================

        void AssertClose(fProxy a, fProxy b, fProxy precision)
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
    }

    public static Array GetEnums() => Enum.GetValues(typeof(TestJob.TestType));

    [TestCaseSource("GetEnums")]
    public void ResampleTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try
        {
            new TestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
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

    // ---------------- Managed validation throws (main thread, not in a Burst job) ----------------

    [Test]
    public void SampleAtEmptyThrows()
    {
        var empty = new fProxyN(0, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Resample.sampleAt(in empty, (fProxy)0, Interp.Linear, EdgeMode.Clamp));
    }

    [Test]
    public void SampleAtIntoValidates()
    {
        var data = new fProxyN(5, Allocator.Temp);
        for (int i = 0; i < 5; i++) data[i] = (fProxy)i;

        // dest.N != positions.N
        var positions = new fProxyN(3, Allocator.Temp);
        var destBad = new fProxyN(4, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Resample.sampleAtInto(in data, in positions, ref destBad, Interp.Linear, EdgeMode.Clamp));

        // empty data (dest.N == positions.N so it reaches the data check) -> "sampleAtInto:" message
        var emptyData = new fProxyN(0, Allocator.Temp);
        var dest = new fProxyN(3, Allocator.Temp);
        var ex = Assert.Throws<ArgumentException>(
            () => Resample.sampleAtInto(in emptyData, in positions, ref dest, Interp.Linear, EdgeMode.Clamp));
        StringAssert.Contains("sampleAtInto:", ex.Message);
    }

    [Test]
    public void ResampleIntoValidates()
    {
        var src = new fProxyN(4, Allocator.Temp);
        var emptyDst = new fProxyN(0, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Resample.resampleInto(in src, ref emptyDst, Interp.Linear, EdgeMode.Clamp));

        var emptySrc = new fProxyN(0, Allocator.Temp);
        var dst = new fProxyN(4, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Resample.resampleInto(in emptySrc, ref dst, Interp.Linear, EdgeMode.Clamp));
    }

    [Test]
    public void Resample2DIntoValidates()
    {
        var src = new fProxyMxN(3, 3, Allocator.Temp);

        // dst with 0 rows
        var dstNoRows = new fProxyMxN(0, 3, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Resample.resample2DInto(in src, ref dstNoRows, Interp.Linear, EdgeMode.Clamp));

        // dst with 0 cols
        var dstNoCols = new fProxyMxN(3, 0, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Resample.resample2DInto(in src, ref dstNoCols, Interp.Linear, EdgeMode.Clamp));

        // src 0x0 (validated before any scratch allocation)
        var emptySrc = new fProxyMxN(0, 0, Allocator.Temp);
        var dst = new fProxyMxN(2, 2, Allocator.Temp);
        Assert.Throws<ArgumentException>(
            () => Resample.resample2DInto(in emptySrc, ref dst, Interp.Linear, EdgeMode.Clamp));
    }
}
