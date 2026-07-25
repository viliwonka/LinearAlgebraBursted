using BULA;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Round-1 Krylov fusion kernels: each fused primitive
// (Blas.axpyNormSq/xpayNormSq/updateXR/scaledCopy/combine3) is checked against the EXACT unfused
// sequence it replaces in Krylov.fProxy.cs, on random vectors, at sizes that exercise both the
// width-4 SIMD block path and the scalar tail (n not a multiple of 4). axpyNormSq/xpayNormSq/
// updateXR preserve the original accumulation order exactly (same axpy/aypx element order, same
// fProxy4-block vecDot fold for the trailing reduction) -- asserted BIT-IDENTICAL. scaledCopy/
// combine3 replace a division with a precomputed-reciprocal multiply -- rounding-only, asserted
// close to machine precision, not bit-identical (user ruling: determinism required, bit-exactness
// not required).
public class fProxyKrylovFusedKernelTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct KrylovFusedKernelTestJob : IJob
    {
        public enum TestType
        {
            AxpyNormSqBitIdentical,
            XpayNormSqBitIdentical,
            UpdateXRBitIdenticalSquare,
            UpdateXRBitIdenticalRectangular,
            ScaledCopyCloseToDivide,
            Combine3CloseToUnfusedChain,
        }

        public TestType Type;

        static readonly int[] Sizes = { 1, 4, 7, 20, 257 };

        // Parallel arrays instead of a ValueTuple[] -- declared static (like Sizes above), not
        // constructed inline in a method body: Burst does not support creating a managed array at
        // runtime inside a job (BC1028), but a statically-initialized readonly array reference compiles.
        static readonly int[] ShapesNx = { 3, 9, 4, 7, 1, 20 };
        static readonly int[] ShapesNr = { 9, 3, 7, 4, 20, 1 };

        // Reciprocal-multiply-vs-divide rounding gap is at most a couple ULPs; use a tight relative
        // tolerance (not zero) for the two kernels that are documented as rounding-only.
        static fProxy RoundingTol() => /*+choose[1e-5f|1e-12]*/1e-5f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.AxpyNormSqBitIdentical: AxpyNormSqBitIdentical(); break;
                case TestType.XpayNormSqBitIdentical: XpayNormSqBitIdentical(); break;
                case TestType.UpdateXRBitIdenticalSquare: UpdateXRBitIdenticalSquare(); break;
                case TestType.UpdateXRBitIdenticalRectangular: UpdateXRBitIdenticalRectangular(); break;
                case TestType.ScaledCopyCloseToDivide: ScaledCopyCloseToDivide(); break;
                case TestType.Combine3CloseToUnfusedChain: Combine3CloseToUnfusedChain(); break;
            }
        }

        static void AssertExact(in fProxyN a, in fProxyN b)
        {
            Assert.AreEqual(a.N, b.N);
            for (int i = 0; i < a.N; i++)
                Assert.AreEqual((double)a[i], (double)b[i]);
        }

        // ---- axpyNormSq: y += a*x ; return dot(y,y) -- vs axpy(y,x,a,n) then Blas.dot(y,y) ----
        void AxpyNormSqBitIdentical()
        {
            foreach (int n in Sizes)
            {
                var x = GenerateOP.fProxyRandomVec(n, -3f, 3f, (uint)(1000 + n));
                var yFused = GenerateOP.fProxyRandomVec(n, -2f, 2f, (uint)(2000 + n));
                var yRef = new fProxyN(n, Allocator.Temp); yRef.Data.CopyFrom(yFused.Data);
                fProxy a = (fProxy)0.37f;

                fProxy fusedNormSq = Blas.axpyNormSq(a, x, ref yFused);

                yRef.addScaledInPlace(a, x);          // the exact call site sequence this replaces
                fProxy refNormSq = Blas.dot(yRef, yRef);

                AssertExact(in yFused, in yRef);
                Assert.AreEqual((double)refNormSq, (double)fusedNormSq);
            }
        }

        // ---- xpayNormSq: y = a*y + x ; return dot(y,y) -- vs aypx(y,x,a,n) then Blas.dot(y,y) ----
        void XpayNormSqBitIdentical()
        {
            foreach (int n in Sizes)
            {
                var x = GenerateOP.fProxyRandomVec(n, -3f, 3f, (uint)(3000 + n));
                var yFused = GenerateOP.fProxyRandomVec(n, -2f, 2f, (uint)(4000 + n));
                var yRef = new fProxyN(n, Allocator.Temp); yRef.Data.CopyFrom(yFused.Data);
                fProxy a = (fProxy)(-0.61f);

                fProxy fusedNormSq = Blas.xpayNormSq(a, x, ref yFused);

                yRef.scaleAddInPlace(a, x);            // the exact call site sequence this replaces
                fProxy refNormSq = Blas.dot(yRef, yRef);

                AssertExact(in yFused, in yRef);
                Assert.AreEqual((double)refNormSq, (double)fusedNormSq);
            }
        }

        // ---- updateXR, square case (cg): x += a*p ; r -= a*q ; return dot(r,r) ----
        void UpdateXRBitIdenticalSquare()
        {
            foreach (int n in Sizes)
            {
                var p = GenerateOP.fProxyRandomVec(n, -1f, 1f, (uint)(5000 + n));
                var q = GenerateOP.fProxyRandomVec(n, -1f, 1f, (uint)(6000 + n));
                var xFused = GenerateOP.fProxyRandomVec(n, -1f, 1f, (uint)(7000 + n));
                var rFused = GenerateOP.fProxyRandomVec(n, -1f, 1f, (uint)(8000 + n));
                var xRef = new fProxyN(n, Allocator.Temp); xRef.Data.CopyFrom(xFused.Data);
                var rRef = new fProxyN(n, Allocator.Temp); rRef.Data.CopyFrom(rFused.Data);
                fProxy a = (fProxy)0.83f;

                fProxy fusedNormSq = Blas.updateXR(a, p, ref xFused, q, ref rFused);

                xRef.addScaledInPlace(a, p);
                rRef.addScaledInPlace(-a, q);
                fProxy refNormSq = Blas.dot(rRef, rRef);

                AssertExact(in xFused, in xRef);
                AssertExact(in rFused, in rRef);
                Assert.AreEqual((double)refNormSq, (double)fusedNormSq);
            }
        }

        // ---- updateXR, RECTANGULAR case (lsqr): x/p length != r/q length. Regression guard:
        // x/p and r/q must use INDEPENDENT lengths, not a shared loop bound. ----
        void UpdateXRBitIdenticalRectangular()
        {
            for (int shape = 0; shape < ShapesNx.Length; shape++)
            {
                int nx = ShapesNx[shape], nr = ShapesNr[shape];
                var p = GenerateOP.fProxyRandomVec(nx, -1f, 1f, (uint)(9000 + nx * 31 + nr));
                var q = GenerateOP.fProxyRandomVec(nr, -1f, 1f, (uint)(9500 + nx * 31 + nr));
                var xFused = GenerateOP.fProxyRandomVec(nx, -1f, 1f, (uint)(10000 + nx * 31 + nr));
                var rFused = GenerateOP.fProxyRandomVec(nr, -1f, 1f, (uint)(10500 + nx * 31 + nr));
                var xRef = new fProxyN(nx, Allocator.Temp); xRef.Data.CopyFrom(xFused.Data);
                var rRef = new fProxyN(nr, Allocator.Temp); rRef.Data.CopyFrom(rFused.Data);
                fProxy a = (fProxy)0.44f;

                fProxy fusedNormSq = Blas.updateXR(a, p, ref xFused, q, ref rFused);

                xRef.addScaledInPlace(a, p);
                rRef.addScaledInPlace(-a, q);
                fProxy refNormSq = Blas.dot(rRef, rRef);

                AssertExact(in xFused, in xRef);
                AssertExact(in rFused, in rRef);
                Assert.AreEqual((double)refNormSq, (double)fusedNormSq);
            }
        }

        // ---- scaledCopy: y = a*x, a = 1/s precomputed -- vs CopyFrom + divInPlace(s) (MINRES's v update) ----
        void ScaledCopyCloseToDivide()
        {
            foreach (int n in Sizes)
            {
                var x = GenerateOP.fProxyRandomVec(n, -5f, 5f, (uint)(11000 + n));
                fProxy s = (fProxy)2.75f;

                var yFused = new fProxyN(n, Allocator.Temp);
                Blas.scaledCopy(1 / s, x, ref yFused);

                var yRef = new fProxyN(n, Allocator.Temp);
                yRef.Data.CopyFrom(x.Data);
                yRef.divInPlace(s);

                fProxyKrylovTestAsserts.AssertVecClose(in yFused, in yRef, RoundingTol());
            }
        }

        // ---- combine3: w = s*(v + a*w1 + b*w2), s = 1/gamma -- vs copy+axpy+axpy+divInPlace (MINRES's w update) ----
        void Combine3CloseToUnfusedChain()
        {
            foreach (int n in Sizes)
            {
                var v = GenerateOP.fProxyRandomVec(n, -1f, 1f, (uint)(12000 + n));
                var w1 = GenerateOP.fProxyRandomVec(n, -1f, 1f, (uint)(13000 + n));
                var w2 = GenerateOP.fProxyRandomVec(n, -1f, 1f, (uint)(14000 + n));
                fProxy a = (fProxy)(-0.29f), b = (fProxy)(-0.53f), gamma = (fProxy)1.9f;

                var wFused = new fProxyN(n, Allocator.Temp);
                Blas.combine3(ref wFused, v, a, w1, b, w2, 1 / gamma);

                var wRef = new fProxyN(n, Allocator.Temp);
                wRef.Data.CopyFrom(v.Data);
                wRef.addScaledInPlace(a, w1);
                wRef.addScaledInPlace(b, w2);
                wRef.divInPlace(gamma);

                fProxyKrylovTestAsserts.AssertVecClose(in wFused, in wRef, RoundingTol());
            }
        }
    }

    [Test]
    public void AxpyNormSqBitIdenticalTest()
        => new KrylovFusedKernelTestJob { Type = KrylovFusedKernelTestJob.TestType.AxpyNormSqBitIdentical }.Run();

    [Test]
    public void XpayNormSqBitIdenticalTest()
        => new KrylovFusedKernelTestJob { Type = KrylovFusedKernelTestJob.TestType.XpayNormSqBitIdentical }.Run();

    [Test]
    public void UpdateXRBitIdenticalSquareTest()
        => new KrylovFusedKernelTestJob { Type = KrylovFusedKernelTestJob.TestType.UpdateXRBitIdenticalSquare }.Run();

    [Test]
    public void UpdateXRBitIdenticalRectangularTest()
        => new KrylovFusedKernelTestJob { Type = KrylovFusedKernelTestJob.TestType.UpdateXRBitIdenticalRectangular }.Run();

    [Test]
    public void ScaledCopyCloseToDivideTest()
        => new KrylovFusedKernelTestJob { Type = KrylovFusedKernelTestJob.TestType.ScaledCopyCloseToDivide }.Run();

    [Test]
    public void Combine3CloseToUnfusedChainTest()
        => new KrylovFusedKernelTestJob { Type = KrylovFusedKernelTestJob.TestType.Combine3CloseToUnfusedChain }.Run();
}
