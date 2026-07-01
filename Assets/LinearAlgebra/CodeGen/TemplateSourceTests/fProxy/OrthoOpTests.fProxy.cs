using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class fProxyOrthoOpTests
{
    [BurstCompile]
    public struct AssemblyTestJob : IJob
    {
        public void Execute()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 16;

            var Q = arena.fProxyRandomMat(dim*2, dim);
            var R = arena.fProxyMat(dim);

            QR.qrDecomposition(ref Q, ref R);

            arena.Dispose();
        }
    }

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            QRDecompIdentity,
            QRDecompIdentityNonSquare,
            QRDecompRandomDiagonal,
            QRDecompRandom,
            QRDecompRandomLarge,
            QRDecompHilbert,
            QRDecompPermutation,
            QRDecompZero,
            QRDecompRankDeficient,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch(Type)
            {
                case TestType.QRDecompIdentity:
                    QRDecompIdentity();
                break;
                case TestType.QRDecompIdentityNonSquare:
                    QRDecompIdentityNonSquare();
                break;
                case TestType.QRDecompRandomDiagonal:
                    QRDecompRandomDiagonal();
                break;
                case TestType.QRDecompRandom:
                    QRDecompRandom();
                break;
                case TestType.QRDecompRandomLarge:
                    QRDecompRandomLarge();
                    break;
                case TestType.QRDecompHilbert:
                    QRDecompHilbert();
                break;
                case TestType.QRDecompPermutation:
                    QRDecompPermutation();
                    break;
                case TestType.QRDecompZero:
                    QRDecompZero();
                    break;
                case TestType.QRDecompRankDeficient:
                    QRDecompRankDeficient();
                    break;
            }
        }

        public void QRDecompIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var Q = arena.fProxyIdentityMat(dim);
            var R = arena.fProxyMat(dim);

            var A = Q.Copy();

            QR.qrDecomposition(ref Q, ref R);

            //Print.Log(A);
            //Print.Log(Q);
            //Print.Log(R);

            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        public void QRDecompIdentityNonSquare()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var Q = arena.fProxyMat(dim*2, dim);
            var R = arena.fProxyMat(dim);

            for(int i = 0; i < dim; i++)
                Q[i, i] = 1f;

            var A = Q.Copy();

            QR.qrDecomposition(ref Q, ref R);


            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        public void QRDecompRandomDiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var Q = arena.fProxyRandomDiagonalMat(dim, 1f, 3f);
            var R = arena.fProxyMat(dim);

            var A = Q.Copy();

            QR.qrDecomposition(ref Q, ref R);

            /*Print.Log(A);
            Print.Log(Q);
            Print.Log(R);*/

            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        public void QRDecompRandom()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var R = arena.fProxyMat(dim);
            var Q = arena.fProxyRandomMat(dim*2, dim, -0.5f, 0.5f, 94221);

            var A = Q.Copy();

            QR.qrDecomposition(ref Q, ref R);

            //Print.Log(R);

            AssertQR(in A, in Q, in R, 1E-05f);

            arena.Dispose();
        }

        public void QRDecompRandomLarge()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 512;

            var R = arena.fProxyMat(dim);
            var Q = arena.fProxyRandomMat(dim * 2, dim, -5f, 5f, 9612221);

            var A = Q.Copy();

            QR.qrDecomposition(ref Q, ref R);

            AssertQR(in A, in Q, in R, 1E-03f);

            arena.Dispose();
        }

        public void QRDecompHilbert()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 20;

            var Q = arena.fProxyHilbertMat(dim);
            var R = arena.fProxyMat(dim);

            var A = Q.Copy();

            QR.qrDecomposition(ref Q, ref R);

            //Print.Log(A);
            //Print.Log(Q);
            //Print.Log(R);

            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        public void QRDecompPermutation() {

            var arena = new Arena(Allocator.Persistent);

            int tests = 32;
            int dim = 16;
            var rand = new Unity.Mathematics.Random(24011);

            for (int i = 0; i < tests; i++) {

                int p0 = rand.NextInt(0, dim);
                int p1 = rand.NextInt(0, dim);

                while(p0 == p1) {
                    p1 = rand.NextInt(0, dim);
                }

                var Q = arena.fProxyPermutationMat(dim, p0, p1);

                p0 = rand.NextInt(0, dim);
                p1 = rand.NextInt(0, dim);

                while (p0 == p1) {
                    p1 = rand.NextInt(0, dim);
                }

                Q = Linear_OP.dot(arena.fProxyPermutationMat(dim, p0, p1), Q);

                var R = arena.fProxyMat(dim);

                var A = Q.Copy();

                QR.qrDecomposition(ref Q, ref R);

                //Print.Log(A);
                //Print.Log(Q);
                //Print.Log(R);

                AssertQR(in A, in Q, in R);
            }
            arena.Dispose();
        }

        public void QRDecompZero() {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;

            var Q = arena.fProxyMat(dim, dim);
            var R = arena.fProxyMat(dim);

            var A = Q.Copy();

            QR.qrDecomposition(ref Q, ref R);

            //Print.Log(A);
            //Print.Log(Q);
            //Print.Log(R);

            AssertQR(in A, in Q, in R);

            arena.Dispose();
        }

        // Rank-deficient tall matrix (column 3 == column 0): the DECOMPOSITION must still be valid —
        // Householder QR reconstructs A = Q*R and keeps Q orthogonal / R upper-triangular regardless
        // of rank (it is only the back-substitution SOLVE that is undefined for rank-deficient A, so
        // that is deliberately not exercised here; QRCP / SVD / pivoted-Cholesky cover the solve).
        public void QRDecompRankDeficient()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 5;
            var Q = arena.fProxyRandomMat(m, n, -1f, 1f, 555123);
            for (int r = 0; r < m; r++)
                Q[r, 3] = Q[r, 0]; // make column 3 a duplicate of column 0 -> rank deficient

            var R = arena.fProxyMat(n);
            var A = Q.Copy();

            QR.qrDecomposition(ref Q, ref R);

            AssertQR(in A, in Q, in R, 1E-4f);

            arena.Dispose();
        }

        private void AssertQR(in fProxyMxN A, in fProxyMxN Q, in fProxyMxN R) => AssertQR(in A, in Q, in R, 1E-6f);
        private void AssertQR(in fProxyMxN A, in fProxyMxN Q, in fProxyMxN R, fProxy precision)
        {
            fProxyMxN shouldBeZero = A - Linear_OP.dot(Q, R);

            var zeroError = Analysis_OP.MaxZeroError(shouldBeZero);

            if (Analysis_OP.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            // Fail layout: [1]=zeroError, [2]=precision, [3]=diff
            if (!(zeroError <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis_OP.isZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis_OP.isUpperTriangular(R, precision));
            Assert.IsTrue(Analysis_OP.isOrthogonal(Q, precision));
        }
    }

    [BurstCompile]
    public struct PrecisionReconstructTestJob : IJob {

        public enum TestType {
            Random,
            RandomDiagonal
        }

        public TestType Type;

        public void Execute() {

            var arena = new Arena(Allocator.Persistent);

            int tests = 64;
            fProxy errorSum = 0;

            for (uint i = 0; i < tests; i++) {

                int dim = 32;

                fProxyMxN A;

                if(Type == TestType.RandomDiagonal)
                    A = arena.fProxyRandomDiagonalMat(dim, 1f, 3f, 21410 + i*i + i*7);
                else
                    A = arena.fProxyRandomMat(dim*2, dim, -25f, +25f, 21410 + i*i + i*7);

                var Q = A.Copy();
                var R = arena.fProxyMat(dim);

                QR.qrDecomposition(ref Q, ref R);

                //Print.Log(Q);
                //Print.Log(R);

                errorSum += ErrorCheckQR(in A, in Q, in R);

                arena.Clear();
            }

            fProxy avgError = errorSum / tests;

            arena.Dispose();
        }

        private fProxy ErrorCheckQR(in fProxyMxN A, in fProxyMxN Q, in fProxyMxN R) {

            fProxyMxN shouldBeZero = A - Linear_OP.dot(Q, R);

            if(Analysis_OP.isAnyNan(in shouldBeZero))
                throw new System.Exception("PrecisionReconstructTestJob: NaN detected");

            //Print.Log(shouldBeZero);

            fProxy zeroError = Analysis_OP.MaxZeroError(shouldBeZero);

            return zeroError;
        }
    }

    [BurstCompile(FloatMode = FloatMode.Deterministic, FloatPrecision = FloatPrecision.High)]
    public struct SolveSystemTestJob : IJob {

        public enum TestType {
            SquareFullRank,
            OverdeterminedFullRank,

            SquareFullRankDirect,
            OverdeterminedFullRankDirect,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute() {

            switch(Type) {

                case TestType.SquareFullRank:
                    SquareFullRank();
                break;
                case TestType.OverdeterminedFullRank:
                    OverdeterminedFullRank();
                break;
                case TestType.SquareFullRankDirect:
                    SquareFullRankDirect();
                break;
                case TestType.OverdeterminedFullRankDirect:
                    OverdeterminedFullRankDirect();
                break;
            }
        }

        void SquareFullRank() {

            var arena = new Arena(Allocator.Persistent);

            int systemDim = 128;
            int randomMatTests = 128;
            int randomVecTests = 32;
            fProxy errorSum = 0;

            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                fProxyMxN A = arena.fProxyRandomMat(systemDim, systemDim, -5, +5, 420 + i * 7);

                for(int d = 0; d < systemDim; d++)
                    A[d, d] += 5.1f + 10f*random.NextFProxy();

                var Q = A.Copy();
                var R = arena.fProxyMat(systemDim);

                QR.qrDecomposition(ref Q, ref R);

                for(uint j = 0; j < randomVecTests; j++) {

                    fProxyN xOrig = arena.fProxyRandomVec(systemDim, -25, +25, 1337 + i * i + j * 5);
                    fProxyN b = Linear_OP.dot(A, xOrig);
                    fProxyN y = Linear_OP.dot(b, Q);

                    Solvers.solveUpperTriangular(ref R, ref y);

                    y.subInpl(xOrig);
                    fProxy zeroError = Analysis_OP.MaxZeroError(y);

                    if(Analysis_OP.isAnyNan(in y)) {
                        throw new System.Exception("SolveSystemTestJob: NaN detected");
                    }

                    // per-solve garbage detector (~3x above the worst observed conditioning-tail
                    // error of ~0.21 float); the avg bound below is the actual quality guard
                    AssertBound(zeroError, (fProxy)2000 * Consts.fProxySqrtEps);

                    errorSum += zeroError;
                }
            }

            fProxy avgError = errorSum / (randomMatTests*randomVecTests);

            // average bound, scaled per precision (see Consts.fProxySqrtEps)
            AssertBound(avgError, (fProxy)150 * Consts.fProxySqrtEps);

            arena.Dispose();
        }

        void OverdeterminedFullRank() {


            int sysDimM = 128;
            int sysDimN = 64;
            int randomMatTests = 32;
            int randomVecTests = 16;
            fProxy errorSum = 0;

            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                var arena = new Arena(Allocator.Persistent);
                fProxyMxN A = arena.fProxyRandomMat(sysDimM, sysDimN, -5, +5, 420 + i * 7);

                for (int d = 0; d < sysDimN; d++)
                    A[d, d] += 5.1f + 10f * random.NextFProxy();

                var Q = A.Copy();
                var R = arena.fProxyMat(sysDimN);

                QR.qrDecomposition(ref Q, ref R);

                for (uint j = 0; j < randomVecTests; j++) {

                    fProxyN xOrig = arena.fProxyRandomVec(sysDimN, -25, +25, 1337 + i * i + j * 5);
                    fProxyN b = Linear_OP.dot(A, xOrig);
                    fProxyN y = Linear_OP.dot(b, Q);

                    Solvers.solveUpperTriangular(ref R, ref y);

                    y.subInpl(xOrig);
                    fProxy zeroError = Analysis_OP.MaxZeroError(y);

                    if (Analysis_OP.isAnyNan(in y)) {
                        throw new System.Exception("SolveSystemTestJob: NaN detected");
                    }

                    // per-solve garbage detector (~3x above the worst observed conditioning-tail
                    // error of ~0.21 float); the avg bound below is the actual quality guard
                    AssertBound(zeroError, (fProxy)2000 * Consts.fProxySqrtEps);

                    errorSum += zeroError;
                }
                arena.Dispose();
            }

            fProxy avgError = errorSum / (randomMatTests * randomVecTests);

            // average bound, scaled per precision (see Consts.fProxySqrtEps)
            AssertBound(avgError, (fProxy)150 * Consts.fProxySqrtEps);
        }

        void SquareFullRankDirect() {

            var arena = new Arena(Allocator.Persistent);

            int systemDim = 128;
            int randomMatTests = 128;
            fProxy errorSum = 0;

            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                fProxyMxN A = arena.fProxyRandomMat(systemDim, systemDim, -5, +5, 420 + i * 7);

                for (int d = 0; d < systemDim; d++)
                    A[d, d] += 5.1f + 10f * random.NextFProxy();

                fProxyN xOrig = arena.fProxyRandomVec(systemDim, -25, +25, 1337 + i * i + i * 5);
                fProxyN b = Linear_OP.dot(A, xOrig);
                fProxyN x = arena.fProxyVec(systemDim);

                QR.qrDirectSolve(ref A, ref b, ref x);

                if (Analysis_OP.isAnyNan(in x)) {
                    throw new System.Exception("SolveSystemTestJob: NaN detected");
                }
                x.subInpl(xOrig);

                fProxy zeroError = Analysis_OP.MaxZeroError(x);

                // per-solve garbage detector (~3x above the worst observed conditioning-tail
                // error of ~0.21 float); the avg bound below is the actual quality guard
                AssertBound(zeroError, (fProxy)2000 * Consts.fProxySqrtEps);

                errorSum += zeroError;

                arena.Clear();
            }

            fProxy avgError = errorSum / (randomMatTests);

            // average bound, scaled per precision (see Consts.fProxySqrtEps)
            AssertBound(avgError, (fProxy)150 * Consts.fProxySqrtEps);

            arena.Dispose();
        }

        void OverdeterminedFullRankDirect() {


            int sysDimM = 128;
            int sysDimN = 64;
            int randomMatTests = 512;
            fProxy errorSum = 0;

            var random = new Unity.Mathematics.Random(1111);

            for (uint i = 0; i < randomMatTests; i++) {

                var arena = new Arena(Allocator.Persistent);
                fProxyMxN A = arena.fProxyRandomMat(sysDimM, sysDimN, -5, +5, 420 + i * 7);

                for (int d = 0; d < sysDimN; d++)
                    A[d, d] += 5.1f + 10f * random.NextFProxy();

                fProxyN xOrig = arena.fProxyRandomVec(sysDimN, -25, +25, 1337 + i * i + i * 5);
                fProxyN b = Linear_OP.dot(A, xOrig);
                fProxyN x = arena.fProxyVec(sysDimN);

                QR.qrDirectSolve(ref A, ref b, ref x);

                if (Analysis_OP.isAnyNan(in x)) {
                    throw new System.Exception("SolveSystemTestJob: NaN detected");
                }

                x.subInpl(xOrig);

                fProxy zeroError = Analysis_OP.MaxZeroError(x);

                // per-solve garbage detector (~3x above the worst observed conditioning-tail
                // error of ~0.21 float); the avg bound below is the actual quality guard
                AssertBound(zeroError, (fProxy)2000 * Consts.fProxySqrtEps);

                errorSum += zeroError;
                arena.Dispose();
            }

            fProxy avgError = errorSum / (randomMatTests);

            // average bound, scaled per precision (see Consts.fProxySqrtEps)
            AssertBound(avgError, (fProxy)150 * Consts.fProxySqrtEps);
        }

        // Fail layout: [0]=flag, [1]=value, [2]=limit, [3]=excess (value - limit)
        private void AssertBound(fProxy value, fProxy limit)
        {
            if (!(value < limit) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = value;
                Fail[2] = limit;
                Fail[3] = value - limit;
            }
            Assert.IsTrue(value < limit);
        }
    }

    public static Array GetEnums() {
        return Enum.GetValues(typeof(TestJob.TestType));
    }

    [TestCaseSource("GetEnums")]
    public void QRDecompTests(TestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try {
            new TestJob() { Type = type, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    [Test]
    public void QRDecompErrorBenchRandom() {
        new PrecisionReconstructTestJob() { Type = PrecisionReconstructTestJob.TestType.Random }.Run();
    }

    [Test]
    public void QRDecompErrorBenchDiagonal() {
        new PrecisionReconstructTestJob() { Type = PrecisionReconstructTestJob.TestType.RandomDiagonal }.Run();
    }

    [Test]
    public void QRDecompErrorSolveSquareSystem() {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try {
            new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.SquareFullRank, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"SquareFullRank: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    [Test]
    public void QRDecompErrorSolveOverdeterminedSystem() {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try {
            new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.OverdeterminedFullRank, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"OverdeterminedFullRank: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    [Test]
    public void QRDecompErrorSolveSquareSystemDirect() {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try {
            new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.SquareFullRankDirect, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"SquareFullRankDirect: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    [Test]
    public void QRDecompErrorSolveOverdeterminedSystemDirect() {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try {
            new SolveSystemTestJob() { Type = SolveSystemTestJob.TestType.OverdeterminedFullRankDirect, Fail = fail }.Run();
            // Under Burst a failed in-job assert logs an exception and aborts the job without
            // throwing to the caller - surface the recorded diagnostics here as well.
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");

        }
        catch (Exception e) {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"OverdeterminedFullRankDirect: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    // ---- LQ decomposition tests ----

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LQTestJob : IJob
    {
        public enum TestType
        {
            LQDecompIdentitySquare,
            LQDecompRandomSquare,
            LQDecompRandomWide_4x9,
            LQDecompRandomWide_8x16,
            LQDecompDiagonalWide,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.LQDecompIdentitySquare:    LQDecompIdentitySquare();    break;
                case TestType.LQDecompRandomSquare:      LQDecompRandomSquare();      break;
                case TestType.LQDecompRandomWide_4x9:   LQDecompRandomWide_4x9();   break;
                case TestType.LQDecompRandomWide_8x16:  LQDecompRandomWide_8x16();  break;
                case TestType.LQDecompDiagonalWide:      LQDecompDiagonalWide();      break;
            }
        }

        void LQDecompIdentitySquare()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 8;
            var A    = arena.fProxyIdentityMat(dim);
            var origA = A.Copy();
            var L    = arena.fProxyMat(dim, dim);
            var Q    = arena.fProxyMat(dim, dim);
            LQ.lqDecomposition(ref A, ref L, ref Q);
            AssertLQ(in origA, in L, in Q, 1E-6f);
            arena.Dispose();
        }

        void LQDecompRandomSquare()
        {
            var arena = new Arena(Allocator.Persistent);
            int dim = 8;
            var A    = arena.fProxyRandomMat(dim, dim, -0.5f, 0.5f, 77123);
            var origA = A.Copy();
            var L    = arena.fProxyMat(dim, dim);
            var Q    = arena.fProxyMat(dim, dim);
            LQ.lqDecomposition(ref A, ref L, ref Q);
            AssertLQ(in origA, in L, in Q, 1E-4f);
            arena.Dispose();
        }

        void LQDecompRandomWide_4x9()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 4, n = 9;
            var A    = arena.fProxyRandomMat(m, n, -0.5f, 0.5f, 94221);
            var origA = A.Copy();
            var L    = arena.fProxyMat(m, m);
            var Q    = arena.fProxyMat(m, n);
            LQ.lqDecomposition(ref A, ref L, ref Q);
            AssertLQ(in origA, in L, in Q, 1E-4f);
            arena.Dispose();
        }

        void LQDecompRandomWide_8x16()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 8, n = 16;
            var A    = arena.fProxyRandomMat(m, n, -1f, 1f, 12345);
            var origA = A.Copy();
            var L    = arena.fProxyMat(m, m);
            var Q    = arena.fProxyMat(m, n);
            LQ.lqDecomposition(ref A, ref L, ref Q);
            AssertLQ(in origA, in L, in Q, 1E-4f);
            arena.Dispose();
        }

        void LQDecompDiagonalWide()
        {
            // 4 x 8: leading 4 x 4 block = 2*I, remaining columns = 0
            var arena = new Arena(Allocator.Persistent);
            int m = 4, n = 8;
            var A    = arena.fProxyMat(m, n);
            for (int i = 0; i < m; i++)
                A[i, i] = (fProxy)2;
            var origA = A.Copy();
            var L    = arena.fProxyMat(m, m);
            var Q    = arena.fProxyMat(m, n);
            LQ.lqDecomposition(ref A, ref L, ref Q);
            AssertLQ(in origA, in L, in Q, 1E-4f);
            arena.Dispose();
        }

        // Checks A ≈ L*Q, L lower-triangular, Q has orthonormal rows (QQᵀ = I_m).
        private void AssertLQ(in fProxyMxN A, in fProxyMxN L, in fProxyMxN Q, fProxy precision)
        {
            // 1. Reconstruction: A ≈ L * Q
            fProxyMxN LQProduct = Linear_OP.dot(L, Q);
            fProxyMxN diff = A - LQProduct;

            if (Analysis_OP.isAnyNan(in diff))
                throw new System.Exception("AssertLQ: NaN in reconstruction");

            fProxy reconError = Analysis_OP.MaxZeroError(diff);
            if (!(reconError <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = reconError;
                Fail[2] = precision;
                Fail[3] = reconError - precision;
            }
            Assert.IsTrue(Analysis_OP.isZero(in diff, precision));

            // 2. L is lower-triangular
            Assert.IsTrue(Analysis_OP.isLowerTriangular(L, precision));

            // 3. Q has orthonormal rows: QQᵀ = I_m.
            //    isOrthogonal(Qᵀ) checks (Qᵀ)ᵀ(Qᵀ) = QQᵀ = I_m.
            fProxyMxN Qt = Linear_OP.trans(Q);
            Assert.IsTrue(Analysis_OP.isOrthogonal(in Qt, precision));
        }
    }

    // ---- LQ min-norm solver tests ----

    [BurstCompile(FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LQMinNormTestJob : IJob
    {
        public enum TestType
        {
            KnownSolutionSmall,
            KnownSolutionWide_4x9,
            KnownSolutionWide_8x16,
            ResidualCheck,
        }

        public TestType Type;

        // [0] flag (1 = failure recorded), [1] got, [2] expected/limit, [3] diff
        public NativeArray<fProxy> Fail;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.KnownSolutionSmall:      KnownSolutionSmall();      break;
                case TestType.KnownSolutionWide_4x9:  KnownSolutionWide_4x9();  break;
                case TestType.KnownSolutionWide_8x16: KnownSolutionWide_8x16(); break;
                case TestType.ResidualCheck:            ResidualCheck();            break;
            }
        }

        // Build x_true = Aᵀ c (row-space), b = A x_true; solve and check x ≈ x_true.
        // x_true is the unique min-norm solution because it lies in row(A) and satisfies Ax = b.
        void KnownSolutionSmall()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 3, n = 6;
            var A    = arena.fProxyRandomMat(m, n, -1f, 1f, 11111);
            var c    = arena.fProxyRandomVec(m, -1f, 1f, 22222);
            // x_true = Aᵀ c  (dot(c, A) computes cᵀA = (Aᵀc)ᵀ → same n-vector values)
            var xTrue = arena.fProxyVec(n);
            Linear_OP.dot(in c, in A, ref xTrue);
            // b = A x_true
            var b = arena.fProxyVec(m);
            Linear_OP.dot(in A, in xTrue, ref b);
            // solve
            var x = arena.fProxyVec(n);
            LQ.lqMinNormSolve(ref A, ref b, ref x);
            AssertClose(in x, in xTrue, 1E-4f);
            arena.Dispose();
        }

        void KnownSolutionWide_4x9()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 4, n = 9;
            var A    = arena.fProxyRandomMat(m, n, -1f, 1f, 33333);
            var c    = arena.fProxyRandomVec(m, -1f, 1f, 44444);
            var xTrue = arena.fProxyVec(n);
            Linear_OP.dot(in c, in A, ref xTrue);
            var b = arena.fProxyVec(m);
            Linear_OP.dot(in A, in xTrue, ref b);
            var x = arena.fProxyVec(n);
            LQ.lqMinNormSolve(ref A, ref b, ref x);
            AssertClose(in x, in xTrue, 1E-4f);
            arena.Dispose();
        }

        void KnownSolutionWide_8x16()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 8, n = 16;
            var A    = arena.fProxyRandomMat(m, n, -1f, 1f, 55555);
            var c    = arena.fProxyRandomVec(m, -1f, 1f, 66666);
            var xTrue = arena.fProxyVec(n);
            Linear_OP.dot(in c, in A, ref xTrue);
            var b = arena.fProxyVec(m);
            Linear_OP.dot(in A, in xTrue, ref b);
            var x = arena.fProxyVec(n);
            LQ.lqMinNormSolve(ref A, ref b, ref x);
            AssertClose(in x, in xTrue, 1E-4f);
            arena.Dispose();
        }

        // Verify that A*x ≈ b (residual is small) independently of the known-solution construction.
        void ResidualCheck()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 5, n = 12;
            var A = arena.fProxyRandomMat(m, n, -2f, 2f, 77777);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 88888);
            var x = arena.fProxyVec(n);
            LQ.lqMinNormSolve(ref A, ref b, ref x);
            // residual = A x - b
            var Ax   = arena.fProxyVec(m);
            Linear_OP.dot(in A, in x, ref Ax);
            Ax.subInpl(b);
            fProxy residual = Analysis_OP.MaxZeroError(Ax);
            if (!(residual <= (fProxy)1E-4f) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = residual;
                Fail[2] = (fProxy)1E-4f;
                Fail[3] = residual - (fProxy)1E-4f;
            }
            Assert.IsTrue(residual <= (fProxy)1E-4f);
            arena.Dispose();
        }

        // Checks that every entry of got matches expected within precision.
        private void AssertClose(in fProxyN got, in fProxyN expected, fProxy precision)
        {
            fProxyN diff = got - expected;

            if (Analysis_OP.isAnyNan(in diff))
                throw new System.Exception("AssertClose: NaN detected");

            fProxy err = Analysis_OP.MaxZeroError(diff);
            if (!(err <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = err;
                Fail[2] = precision;
                Fail[3] = err - precision;
            }
            Assert.IsTrue(Analysis_OP.isZero(in diff, precision));
        }
    }

    public static Array GetLQEnums()      => Enum.GetValues(typeof(LQTestJob.TestType));
    public static Array GetLQSolveEnums() => Enum.GetValues(typeof(LQMinNormTestJob.TestType));

    [TestCaseSource("GetLQEnums")]
    public void LQDecompTests(LQTestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try {
            new LQTestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e) {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }

    [TestCaseSource("GetLQSolveEnums")]
    public void LQMinNormSolveTests(LQMinNormTestJob.TestType type)
    {
        var fail = new NativeArray<fProxy>(4, Allocator.TempJob);
        try {
            new LQMinNormTestJob() { Type = type, Fail = fail }.Run();
            if (fail[0] != (fProxy)0)
                Assert.Fail($"got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]}");
        }
        catch (Exception e) {
            if (fail[0] != (fProxy)0)
                Assert.Fail($"{type}: got {fail[1]}, expected/limit {fail[2]}, diff/extra {fail[3]} ({e.Message})");
            throw;
        }
        finally {
            fail.Dispose();
        }
    }
}
