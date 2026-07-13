using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;

using Unity.Jobs;
using UnityEngine;
using UnityEngine.TestTools;

public class fProxyQRTests
{
    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
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
            // Blocked path (N_Cols >= Consts.floatQrBlockMinN=128 / doubleQrBlockMinN=512) with a
            // LAST panel that is NARROWER than QR_BLOCK (=32): N_Cols NOT a multiple of 32. Tall
            // (M_Rows >= N_Cols). Only QRDecompBlockedNonAligned_1100x545 (N_Cols=545) clears the
            // double gate; the smaller shapes below stay on the unblocked fallback for both types.
            QRDecompBlockedNonAligned_200x100,
            QRDecompBlockedNonAligned_130x65,
            QRDecompBlockedNonAligned_150x70,
            QRDecompBlockedNonAligned_200x127,
            QRDecompBlockedNonAligned_160x96,
            QRDecompBlockedNonAligned_256x150,
            QRDecompBlockedNonAligned_1100x545,
            // Solver API rework (commit 2) coverage.
            QRDecompPreservesA,
            QRUninitXContract,
            // Commit 2.5 (2f-i): QR.decomp A-preservation at the BLOCKED-path scale (N_Cols >= 64).
            QRDecompPreservesABlocked,
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
                case TestType.QRDecompBlockedNonAligned_200x100:
                    QRDecompBlockedNonAligned(200, 100, 700011);
                    break;
                case TestType.QRDecompBlockedNonAligned_130x65:
                    QRDecompBlockedNonAligned(130, 65, 700065);
                    break;
                case TestType.QRDecompBlockedNonAligned_150x70:
                    QRDecompBlockedNonAligned(150, 70, 700070);
                    break;
                case TestType.QRDecompBlockedNonAligned_200x127:
                    QRDecompBlockedNonAligned(200, 127, 700127);
                    break;
                case TestType.QRDecompBlockedNonAligned_160x96:
                    QRDecompBlockedNonAligned(160, 96, 700096);
                    break;
                case TestType.QRDecompBlockedNonAligned_256x150:
                    QRDecompBlockedNonAligned(256, 150, 700150);
                    break;
                case TestType.QRDecompBlockedNonAligned_1100x545:
                    QRDecompBlockedNonAligned(1100, 545, 700545);
                    break;
                case TestType.QRDecompPreservesA:
                    QRDecompPreservesA();
                    break;
                case TestType.QRUninitXContract:
                    QRUninitXContract();
                    break;
                case TestType.QRDecompPreservesABlocked:
                    QRDecompPreservesABlocked();
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

            QR.decompInPlace(ref Q, ref R);

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

            QR.decompInPlace(ref Q, ref R);


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

            QR.decompInPlace(ref Q, ref R);

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

            QR.decompInPlace(ref Q, ref R);

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

            QR.decompInPlace(ref Q, ref R);

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

            QR.decompInPlace(ref Q, ref R);

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

                Q = Blas.dot(arena.fProxyPermutationMat(dim, p0, p1), Q);

                var R = arena.fProxyMat(dim);

                var A = Q.Copy();

                QR.decompInPlace(ref Q, ref R);

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

            QR.decompInPlace(ref Q, ref R);

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

            QR.decompInPlace(ref Q, ref R);

            AssertQR(in A, in Q, in R, 1E-4f);

            arena.Dispose();
        }

        // Exercises the BLOCKED QR path (engaged per-type when N_Cols >= Consts.floatQrBlockMinN=128
        // / doubleQrBlockMinN=512) at column counts that are NOT multiples of the block width
        // QR_BLOCK (=32), so the trailing panel is narrower than a full block (n mod 32 != 0).
        // Tall shapes (M_Rows >= N_Cols) with a boosted diagonal to stay well-conditioned/full-rank,
        // matching the solver tests. AssertQR verifies all three invariants: A ≈ Q*R (reconstruction),
        // QᵀQ ≈ I (orthonormal columns), R upper-triangular.
        void QRDecompBlockedNonAligned(int m, int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var random = new Unity.Mathematics.Random(seed);
            var Q = arena.fProxyRandomMat(m, n, -5f, 5f, seed);
            for (int d = 0; d < n; d++)
                Q[d, d] += 5.1f + 10f * random.NextFProxy();

            var R = arena.fProxyMat(n);
            var A = Q.Copy();

            QR.decompInPlace(ref Q, ref R);

            AssertQR(in A, in Q, in R, 1E-3f);

            arena.Dispose();
        }

        // Solver API rework (commit 2): QR.decomp must not modify A. Checksum (position-weighted
        // sum, so a permutation or a single altered entry both trip it) before/after the call.
        void QRDecompPreservesA()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 12, n = 6;
            var A = arena.fProxyRandomMat(m, n, -3f, 3f, 909090);
            for (int d = 0; d < n; d++) A[d, d] += 5f;

            fProxy checksumBefore = (fProxy)0;
            for (int i = 0; i < A.Length; i++) checksumBefore += A[i] * (fProxy)(i + 1);

            var Q = arena.fProxyMat(m, n);
            var R = arena.fProxyMat(n);
            QR.decomp(in A, ref Q, ref R);

            fProxy checksumAfter = (fProxy)0;
            for (int i = 0; i < A.Length; i++) checksumAfter += A[i] * (fProxy)(i + 1);

            if (checksumAfter != checksumBefore && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = checksumAfter;
                Fail[2] = checksumBefore;
                Fail[3] = checksumAfter - checksumBefore;
            }
            Assert.IsTrue(checksumAfter == checksumBefore);

            // and the decomposition itself must still be correct (A intact, matches Q*R).
            AssertQR(in A, in Q, in R, 1E-4f);

            arena.Dispose();
        }

        // Uninit-x contract: QR.solveInPlace and QR.decompSolve must treat x as OUTPUT ONLY -- prior
        // garbage (here, NaN sentinels) must not survive into the result.
        void QRUninitXContract()
        {
            var arena = new Arena(Allocator.Persistent);

            int dim = 8;
            var A = arena.fProxyRandomMat(dim, dim, -5f, 5f, 13131);
            for (int d = 0; d < dim; d++) A[d, d] += 10f;
            var xKnown = arena.fProxyRandomVec(dim, -3f, 3f, 24242);
            var b = Blas.dot(A, xKnown);

            // QR.solveInPlace: x pre-filled with NaN.
            {
                var Awork = A.Copy();
                var bwork = b.Copy();
                var x = arena.fProxyVec(dim);
                for (int i = 0; i < dim; i++) x[i] = fProxy.NaN;

                QR.solveInPlace(ref Awork, ref bwork, ref x);

                Assert.IsFalse(Analysis.isAnyNan(in x));
                for (int i = 0; i < dim; i++)
                {
                    fProxy diff = Unity.Mathematics.math.abs(x[i] - xKnown[i]);
                    if (!(diff <= (fProxy)1E-3f) && Fail[0] == (fProxy)0)
                    {
                        Fail[0] = (fProxy)1; Fail[1] = x[i]; Fail[2] = xKnown[i]; Fail[3] = diff;
                    }
                    Assert.IsTrue(diff <= (fProxy)1E-3f);
                }
            }

            // QR.decompSolve: x pre-filled with NaN, Q/R from a fresh decompInPlace.
            {
                var Q = A.Copy();
                var R = arena.fProxyMat(dim);
                QR.decompInPlace(ref Q, ref R);

                var x = arena.fProxyVec(dim);
                for (int i = 0; i < dim; i++) x[i] = fProxy.NaN;
                var bcopy = b.Copy();

                QR.decompSolve(ref Q, ref R, ref bcopy, ref x);

                Assert.IsFalse(Analysis.isAnyNan(in x));
                for (int i = 0; i < dim; i++)
                {
                    fProxy diff = Unity.Mathematics.math.abs(x[i] - xKnown[i]);
                    if (!(diff <= (fProxy)1E-3f) && Fail[0] == (fProxy)0)
                    {
                        Fail[0] = (fProxy)1; Fail[1] = x[i]; Fail[2] = xKnown[i]; Fail[3] = diff;
                    }
                    Assert.IsTrue(diff <= (fProxy)1E-3f);
                }
            }

            arena.Dispose();
        }

        // Blocked-path A-preservation: QR.decomp at N_Cols >= Consts.floatQrBlockMinN=128 /
        // doubleQrBlockMinN=512 engages the level-3 blocked (compact-WY GEMM trailing-update) path;
        // it still must not modify A. Uses a tall well-conditioned 576x512 shape (clears the double
        // gate too, matching QRDecompBlockedNonAligned's construction). The existing QRDecompPreservesA
        // only reaches the unblocked path (12x6).
        void QRDecompPreservesABlocked()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 576, n = 512;
            var random = new Unity.Mathematics.Random(576512);
            var A = arena.fProxyRandomMat(m, n, -5f, 5f, 576512);
            for (int d = 0; d < n; d++)
                A[d, d] += 5.1f + 10f * random.NextFProxy();

            fProxy checksumBefore = (fProxy)0;
            for (int i = 0; i < A.Length; i++) checksumBefore += A[i] * (fProxy)(i + 1);

            var Q = arena.fProxyMat(m, n);
            var R = arena.fProxyMat(n);
            QR.decomp(in A, ref Q, ref R);

            fProxy checksumAfter = (fProxy)0;
            for (int i = 0; i < A.Length; i++) checksumAfter += A[i] * (fProxy)(i + 1);

            if (checksumAfter != checksumBefore && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = checksumAfter;
                Fail[2] = checksumBefore;
                Fail[3] = checksumAfter - checksumBefore;
            }
            Assert.IsTrue(checksumAfter == checksumBefore);

            // decomposition itself must still be correct (A intact, matches Q*R).
            AssertQR(in A, in Q, in R, 1E-3f);

            arena.Dispose();
        }

        private void AssertQR(in fProxyMxN A, in fProxyMxN Q, in fProxyMxN R) => AssertQR(in A, in Q, in R, 1E-6f);
        private void AssertQR(in fProxyMxN A, in fProxyMxN Q, in fProxyMxN R, fProxy precision)
        {
            fProxyMxN shouldBeZero = A - Blas.dot(Q, R);

            var zeroError = Analysis.MaxZeroError(shouldBeZero);

            if (Analysis.isAnyNan(in shouldBeZero))
                throw new System.Exception("TestJob: NaN detected");

            // Fail layout: [1]=zeroError, [2]=precision, [3]=diff
            if (!(zeroError <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = zeroError;
                Fail[2] = precision;
                Fail[3] = zeroError - precision;
            }
            Assert.IsTrue(Analysis.isZero(in shouldBeZero, precision));
            Assert.IsTrue(Analysis.isUpperTriangular(R, precision));
            Assert.IsTrue(Analysis.isOrthogonal(Q, precision));
        }
    }

    [BurstCompile(CompileSynchronously = true)]
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

                QR.decompInPlace(ref Q, ref R);

                errorSum += ErrorCheckQR(in A, in Q, in R);

                arena.Clear();
            }

            fProxy avgError = errorSum / tests;

            arena.Dispose();
        }

        private fProxy ErrorCheckQR(in fProxyMxN A, in fProxyMxN Q, in fProxyMxN R) {

            fProxyMxN shouldBeZero = A - Blas.dot(Q, R);

            if(Analysis.isAnyNan(in shouldBeZero))
                throw new System.Exception("PrecisionReconstructTestJob: NaN detected");

            fProxy zeroError = Analysis.MaxZeroError(shouldBeZero);

            return zeroError;
        }
    }

    [BurstCompile(CompileSynchronously = true, FloatMode = FloatMode.Deterministic, FloatPrecision = FloatPrecision.High)]
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

                QR.decompInPlace(ref Q, ref R);

                for(uint j = 0; j < randomVecTests; j++) {

                    fProxyN xOrig = arena.fProxyRandomVec(systemDim, -25, +25, 1337 + i * i + j * 5);
                    fProxyN b = Blas.dot(A, xOrig);
                    fProxyN y = Blas.dot(b, Q);

                    Blas.triUpper(ref R, ref y);

                    y.subInPlace(xOrig);
                    fProxy zeroError = Analysis.MaxZeroError(y);

                    if(Analysis.isAnyNan(in y)) {
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

                QR.decompInPlace(ref Q, ref R);

                for (uint j = 0; j < randomVecTests; j++) {

                    fProxyN xOrig = arena.fProxyRandomVec(sysDimN, -25, +25, 1337 + i * i + j * 5);
                    fProxyN b = Blas.dot(A, xOrig);
                    fProxyN y = Blas.dot(b, Q);

                    Blas.triUpper(ref R, ref y);

                    y.subInPlace(xOrig);
                    fProxy zeroError = Analysis.MaxZeroError(y);

                    if (Analysis.isAnyNan(in y)) {
                        throw new System.Exception("SolveSystemTestJob: NaN detected");
                    }

                    // per-solve bound: see SquareFullRank's rationale above
                    AssertBound(zeroError, (fProxy)2000 * Consts.fProxySqrtEps);

                    errorSum += zeroError;
                }
                arena.Dispose();
            }

            fProxy avgError = errorSum / (randomMatTests * randomVecTests);

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
                fProxyN b = Blas.dot(A, xOrig);
                fProxyN x = arena.fProxyVec(systemDim);

                QR.solveInPlace(ref A, ref b, ref x);

                if (Analysis.isAnyNan(in x)) {
                    throw new System.Exception("SolveSystemTestJob: NaN detected");
                }
                x.subInPlace(xOrig);

                fProxy zeroError = Analysis.MaxZeroError(x);

                // per-solve bound: see SquareFullRank's rationale above
                AssertBound(zeroError, (fProxy)2000 * Consts.fProxySqrtEps);

                errorSum += zeroError;

                arena.Clear();
            }

            fProxy avgError = errorSum / (randomMatTests);

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
                fProxyN b = Blas.dot(A, xOrig);
                fProxyN x = arena.fProxyVec(sysDimN);

                QR.solveInPlace(ref A, ref b, ref x);

                if (Analysis.isAnyNan(in x)) {
                    throw new System.Exception("SolveSystemTestJob: NaN detected");
                }

                x.subInPlace(xOrig);

                fProxy zeroError = Analysis.MaxZeroError(x);

                // per-solve bound: see SquareFullRank's rationale above
                AssertBound(zeroError, (fProxy)2000 * Consts.fProxySqrtEps);

                errorSum += zeroError;
                arena.Dispose();
            }

            fProxy avgError = errorSum / (randomMatTests);

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
            // Burst in-job asserts abort without throwing; diagnostics surfaced here too (see QRDecompTests above).
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
            // Burst in-job asserts abort without throwing; diagnostics surfaced here too (see QRDecompTests above).
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
            // Burst in-job asserts abort without throwing; diagnostics surfaced here too (see QRDecompTests above).
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
            // Burst in-job asserts abort without throwing; diagnostics surfaced here too (see QRDecompTests above).
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

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LQTestJob : IJob
    {
        public enum TestType
        {
            LQDecompIdentitySquare,
            LQDecompRandomSquare,
            LQDecompRandomWide_4x9,
            LQDecompRandomWide_8x16,
            LQDecompDiagonalWide,
            // BLOCKED path (engaged only when M_Rows >= LQ_BLOCK_MIN_M = 512; below that
            // lqDecomposition falls back to the unblocked lqKernel). LQ_BLOCK = 64, so to also
            // exercise the "last panel narrower than LQ_BLOCK" branch we include m values that are
            // NOT multiples of 64 (513 -> pb=1, 700 -> pb=60) alongside exact multiples (512, 576,
            // 640). Wide-or-square shapes (M_Rows <= N_Cols), boosted diagonal for full-row-rank
            // conditioning, matching the QR blocked-non-aligned tests. These are the ONLY tests that
            // reach the blocked LQ core.
            LQDecompBlockedAligned_512x1024,
            LQDecompBlockedLastPanel_513x1030,
            LQDecompBlockedNearSquare_700x701,
            LQDecompBlockedAligned_576x1200,
            LQDecompBlockedNearSquare_640x641,
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
                case TestType.LQDecompBlockedAligned_512x1024:  LQDecompBlockedRandom(512, 1024, 800512); break;
                case TestType.LQDecompBlockedLastPanel_513x1030: LQDecompBlockedRandom(513, 1030, 800513); break;
                case TestType.LQDecompBlockedNearSquare_700x701: LQDecompBlockedRandom(700, 701, 800700); break;
                case TestType.LQDecompBlockedAligned_576x1200:  LQDecompBlockedRandom(576, 1200, 800576); break;
                case TestType.LQDecompBlockedNearSquare_640x641: LQDecompBlockedRandom(640, 641, 800640); break;
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
            LQ.decomp(in A, ref L, ref Q);
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
            LQ.decomp(in A, ref L, ref Q);
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
            LQ.decomp(in A, ref L, ref Q);
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
            LQ.decomp(in A, ref L, ref Q);
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
            LQ.decomp(in A, ref L, ref Q);
            AssertLQ(in origA, in L, in Q, 1E-4f);
            arena.Dispose();
        }

        // Exercises the BLOCKED LQ path (engaged only when m = M_Rows >= LQ_BLOCK_MIN_M = 512; the
        // whole existing suite tops out at 8x16 and so only ever hits the unblocked lqKernel). Uses
        // wide-or-square shapes (m <= n) with a boosted diagonal so A stays full-row-rank and well
        // conditioned — exactly as the QR blocked-non-aligned tests do. AssertLQ checks all three
        // invariants: A ≈ L*Q (reconstruction), L lower-triangular, Q has orthonormal ROWS (QQᵀ=I_m).
        // Tolerance is scale-appropriate for these large sizes (see AssertLQ callsite): at m up to
        // 700 with entries up to ~15 the float reconstruction error is ~1e-3 absolute; a tiny 1e-4
        // bound would false-fail purely on float rounding, so 1e-2 is used (double is far tighter but
        // shares the same bound). A genuine blocked-core bug (non-triangular L, non-orthonormal Q, or
        // reconstruction off by O(1)) still trips this bound loudly.
        void LQDecompBlockedRandom(int m, int n, uint seed)
        {
            var arena = new Arena(Allocator.Persistent);

            var random = new Unity.Mathematics.Random(seed);
            var A = arena.fProxyRandomMat(m, n, -5f, 5f, seed);
            for (int d = 0; d < m; d++)
                A[d, d] += 5.1f + 10f * random.NextFProxy();

            var origA = A.Copy();
            var L = arena.fProxyMat(m, m);
            var Q = arena.fProxyMat(m, n);

            LQ.decomp(in A, ref L, ref Q);

            AssertLQ(in origA, in L, in Q, 1E-2f);

            arena.Dispose();
        }

        // Checks A ≈ L*Q, L lower-triangular, Q has orthonormal rows (QQᵀ = I_m).
        private void AssertLQ(in fProxyMxN A, in fProxyMxN L, in fProxyMxN Q, fProxy precision)
        {
            // 1. Reconstruction: A ≈ L * Q
            fProxyMxN LQProduct = Blas.dot(L, Q);
            fProxyMxN diff = A - LQProduct;

            if (Analysis.isAnyNan(in diff))
                throw new System.Exception("AssertLQ: NaN in reconstruction");

            fProxy reconError = Analysis.MaxZeroError(diff);
            if (!(reconError <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = reconError;
                Fail[2] = precision;
                Fail[3] = reconError - precision;
            }
            Assert.IsTrue(Analysis.isZero(in diff, precision));

            // 2. L is lower-triangular
            Assert.IsTrue(Analysis.isLowerTriangular(L, precision));

            // 3. Q has orthonormal rows: QQᵀ = I_m.
            //    isOrthogonal(Qᵀ) checks (Qᵀ)ᵀ(Qᵀ) = QQᵀ = I_m.
            fProxyMxN Qt = Blas.trans(Q);
            Assert.IsTrue(Analysis.isOrthogonal(in Qt, precision));
        }
    }

    // ---- LQ min-norm solver tests ----

    [BurstCompile(CompileSynchronously = true, FloatPrecision = FloatPrecision.High, FloatMode = FloatMode.Default)]
    public struct LQMinNormTestJob : IJob
    {
        public enum TestType
        {
            KnownSolutionSmall,
            KnownSolutionWide_4x9,
            KnownSolutionWide_8x16,
            ResidualCheck,
            // Large m (>= LQ_BLOCK_MIN_M): exercises the blocked factor-only path in lqFactorInPlace.
            ResidualCheckLargeBlocked,
            // Mid m (256 <= m < 512): the per-type gate split — float takes the BLOCKED core here,
            // double stays UNBLOCKED. Guards that both routes agree at the size where they diverge.
            ResidualCheckMidBlocked,
            // Solver API rework (commit 2): uninit-x contract.
            UninitXContract,
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
                case TestType.ResidualCheckLargeBlocked: ResidualCheckLargeBlocked(); break;
                case TestType.ResidualCheckMidBlocked:   ResidualCheckMidBlocked();   break;
                case TestType.UninitXContract:          UninitXContract();          break;
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
            Blas.dot(in c, in A, ref xTrue);
            // b = A x_true
            var b = arena.fProxyVec(m);
            Blas.dot(in A, in xTrue, ref b);
            // solve
            var x = arena.fProxyVec(n);
            LQ.minNormSolve(ref A, ref b, ref x);
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
            Blas.dot(in c, in A, ref xTrue);
            var b = arena.fProxyVec(m);
            Blas.dot(in A, in xTrue, ref b);
            var x = arena.fProxyVec(n);
            LQ.minNormSolve(ref A, ref b, ref x);
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
            Blas.dot(in c, in A, ref xTrue);
            var b = arena.fProxyVec(m);
            Blas.dot(in A, in xTrue, ref b);
            var x = arena.fProxyVec(n);
            LQ.minNormSolve(ref A, ref b, ref x);
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
            LQ.minNormSolve(ref A, ref b, ref x);
            // residual = A x - b
            var Ax   = arena.fProxyVec(m);
            Blas.dot(in A, in x, ref Ax);
            Ax.subInPlace(b);
            fProxy residual = Analysis.MaxZeroError(Ax);
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

        // Same residual check but at m >= LQ_BLOCK_MIN_M so lqFactorInPlace takes the blocked
        // (compact-WY, level-3) factor-only branch instead of the small unblocked kernel. Leading
        // diagonal is boosted for conditioning so the residual stays tight at float precision.
        void ResidualCheckLargeBlocked()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 520, n = 640;   // m > 512 = LQ_BLOCK_MIN_M -> blocked path
            var A = arena.fProxyRandomMat(m, n, -2f, 2f, 131313);
            for (int d = 0; d < m; d++)
                A[d, d] += (fProxy)20f;
            var b = arena.fProxyRandomVec(m, -1f, 1f, 141414);
            var x = arena.fProxyVec(n);
            LQ.minNormSolve(ref A, ref b, ref x);
            var Ax = arena.fProxyVec(m);
            Blas.dot(in A, in x, ref Ax);
            Ax.subInPlace(b);
            fProxy residual = Analysis.MaxZeroError(Ax);
            fProxy tol = (fProxy)1E-3f;   // larger system, looser float tolerance
            if (!(residual <= tol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = residual;
                Fail[2] = tol;
                Fail[3] = residual - tol;
            }
            Assert.IsTrue(residual <= tol);
            arena.Dispose();
        }

        // Residual check at 256 <= m < 512: float routes to the BLOCKED core (floatLqBlockMinM=256)
        // while double stays on the UNBLOCKED kernel (doubleLqBlockMinM=512). Confirms the split gate
        // routes each type correctly and both produce a valid min-norm solution at the divergence size.
        void ResidualCheckMidBlocked()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 300, n = 400;   // 256 <= m < 512 -> float blocked, double unblocked
            var A = arena.fProxyRandomMat(m, n, -2f, 2f, 151617);
            for (int d = 0; d < m; d++)
                A[d, d] += (fProxy)20f;
            var b = arena.fProxyRandomVec(m, -1f, 1f, 181920);
            var x = arena.fProxyVec(n);
            LQ.minNormSolve(ref A, ref b, ref x);
            var Ax = arena.fProxyVec(m);
            Blas.dot(in A, in x, ref Ax);
            Ax.subInPlace(b);
            fProxy residual = Analysis.MaxZeroError(Ax);
            fProxy tol = (fProxy)1E-3f;
            if (!(residual <= tol) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = residual;
                Fail[2] = tol;
                Fail[3] = residual - tol;
            }
            Assert.IsTrue(residual <= tol);
            arena.Dispose();
        }

        // Uninit-x contract: LQ.minNormSolve must treat x as OUTPUT ONLY -- prior garbage (here, NaN
        // sentinels) must not survive into the result.
        void UninitXContract()
        {
            var arena = new Arena(Allocator.Persistent);
            int m = 5, n = 12;
            var A = arena.fProxyRandomMat(m, n, -2f, 2f, 191919);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 292929);

            var x = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) x[i] = fProxy.NaN;

            LQ.minNormSolve(ref A, ref b, ref x);

            Assert.IsFalse(Analysis.isAnyNan(in x));

            var Ax = arena.fProxyVec(m);
            Blas.dot(in A, in x, ref Ax);
            Ax.subInPlace(b);
            fProxy residual = Analysis.MaxZeroError(Ax);
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

            if (Analysis.isAnyNan(in diff))
                throw new System.Exception("AssertClose: NaN detected");

            fProxy err = Analysis.MaxZeroError(diff);
            if (!(err <= precision) && Fail[0] == (fProxy)0)
            {
                Fail[0] = (fProxy)1;
                Fail[1] = err;
                Fail[2] = precision;
                Fail[3] = err - precision;
            }
            Assert.IsTrue(Analysis.isZero(in diff, precision));
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
