using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Right (column) preconditioned least-squares entry points Krylov.lsqrRightPre / lsmrRightPre:
// a general NON-DIAGONAL symmetric-positive-definite N supplied through IfProxyPreconditioner
// must converge to the same least-squares solution as a direct dense QR solve (elementwise), with
// the exact recomputed optimality residual ||A^T r|| small -- over dense AND BSR A. The *Jacobi
// wrappers (the diagonal case of the same path) must keep matching the QR oracle.
public class fProxyLstsqRightPrecondTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct TestJob : IJob
    {
        public enum TestType
        {
            LsqrRightPreSpdDense,
            LsmrRightPreSpdDense,
            LsqrRightPreSpdBSR,
            LsmrRightPreSpdBSR,
            JacobiWrappersMatchOracle,
            LsqrRightPreOpRinvDense,
            LsmrRightPreOpRinvDense,
            LsqrRightPreOpNonSymDense,
            LsqrRightPreOpNonSymBSR,
        }

        public TestType Type;

        // Elementwise agreement band vs the QR oracle (well-conditioned random systems) --
        // mirrors KrylovLstsqBatteryTests' WellConditioned band.
        static fProxy Band() => (fProxy)50 * Consts.fProxySqrtEps;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.LsqrRightPreSpdDense:       LsqrRightPreSpdDense();       break;
                case TestType.LsmrRightPreSpdDense:       LsmrRightPreSpdDense();       break;
                case TestType.LsqrRightPreSpdBSR:         LsqrRightPreSpdBSR();         break;
                case TestType.LsmrRightPreSpdBSR:         LsmrRightPreSpdBSR();         break;
                case TestType.JacobiWrappersMatchOracle:  JacobiWrappersMatchOracle();  break;
                case TestType.LsqrRightPreOpRinvDense:    LsqrRightPreOpRinvDense();    break;
                case TestType.LsmrRightPreOpRinvDense:    LsmrRightPreOpRinvDense();    break;
                case TestType.LsqrRightPreOpNonSymDense:  LsqrRightPreOpNonSymDense();  break;
                case TestType.LsqrRightPreOpNonSymBSR:    LsqrRightPreOpNonSymBSR();    break;
            }
        }

        // ---- helpers ----

        // R^-1 (n x n dense) from the thin QR of A (A full column rank): A·R^-1 = Q has orthonormal
        // columns, so it is the canonical STRONG least-squares right preconditioner (Blendenpik/LSRN
        // shape) -- and non-symmetric (upper-triangular). Built by upper-triangular back-substitution
        // solving R·Rinv = I column by column.
        static fProxyMxN BuildRinv(in fProxyMxN A)
        {
            int n = A.N_Cols;
            var Q = new fProxyMxN(A.M_Rows, n, Allocator.Temp);
            var R = new fProxyMxN(n, n, Allocator.Temp);
            QR.decomp(in A, ref Q, ref R);
            var Rinv = new fProxyMxN(n, n, Allocator.Temp);
            for (int c = 0; c < n; c++)
                for (int i = n - 1; i >= 0; i--)
                {
                    fProxy s = (i == c) ? (fProxy)1 : (fProxy)0;
                    for (int k = i + 1; k < n; k++) s -= R[i, k] * Rinv[k, c];
                    Rinv[i, c] = s / R[i, i];
                }
            return Rinv;
        }

        // A genuinely non-symmetric, well-conditioned invertible N = I + 0.25·(strictly-lower random):
        // unit diagonal over a nilpotent lower part, so every eigenvalue is 1 (invertible) and
        // N != N^T. Exercises the wrapper's N^T (ApplyT) path, which the symmetric path never touches.
        static fProxyMxN BuildNonSym(int n, uint seed)
        {
            var W = GenerateOP.fProxyRandomMat(n, n, -1f, 1f, seed);
            var Nmat = new fProxyMxN(n, n, Allocator.Temp);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Nmat[i, j] = (i == j) ? (fProxy)1 : (j < i ? (fProxy)0.25 * W[i, j] : (fProxy)0);
            return Nmat;
        }

        // Random m x n A with columns geometrically scaled to condition number ~1e2 -- ill-scaled
        // enough that a column preconditioner earns its keep, mild enough to stay inside the float band.
        static fProxyMxN BuildIllScaled(int m, int n, uint seed)
        {
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, seed);
            for (int j = 0; j < n; j++)
            {
                fProxy scale = math.pow((fProxy)10, (fProxy)2 * j / (fProxy)(n - 1));
                for (int i = 0; i < m; i++) A[i, j] *= scale;
            }
            return A;
        }

        // Direct dense least-squares reference via QR (A full column rank, M_Rows >= N_Cols).
        static fProxyN ReferenceSolveLstsq(in fProxyMxN A, in fProxyN b)
        {
            var Q = new fProxyMxN(A.M_Rows, A.N_Cols, Allocator.Temp);
            var R = new fProxyMxN(A.N_Cols, A.N_Cols, Allocator.Temp);
            QR.decomp(in A, ref Q, ref R);
            fProxyN bLocal = b;
            var xRef = new fProxyN(A.N_Cols, Allocator.Temp);
            QR.decompSolve(ref Q, ref R, ref bLocal, ref xRef);
            return xRef;
        }

        // 1x1-block BSR copy of a dense matrix (scalar AddValue per nonzero).
        static fProxyBSR DenseToBSR(in fProxyMxN A)
        {
            var builder = new fProxyBSRBuilder(A.M_Rows, A.N_Cols, 1, 1, Allocator.Temp, A.M_Rows * A.N_Cols);
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSR(Allocator.Temp);
        }

        static void AssertClose(fProxy got, fProxy expected, fProxy tol)
            => Assert.IsTrue(math.abs(got - expected) <= tol * ((fProxy)1 + math.abs(expected)));

        // Shared checks on a solved system: converged, exact recomputed ||A^T r|| small on the
        // fixed ||A^T b|| scale (100x headroom over tol covers cond(N) mapping the y-space
        // stopping test back to original coordinates), elementwise agreement with the QR oracle.
        static void CheckSolution(in fProxyMxN A, in fProxyN b, in fProxyN x, in LstsqInfo info, fProxy tol)
        {
            Assert.IsTrue(info.Solved);

            var atb = new fProxyN(A.N_Cols, Allocator.Temp);
            Blas.dot(in b, in A, ref atb);                    // A^T b
            fProxy atbNorm = math.sqrt(Blas.dot(atb, atb));
            Assert.IsTrue((fProxy)info.Arnorm <= (fProxy)100 * tol * math.max(atbNorm, (fProxy)1e-30));

            var xRef = ReferenceSolveLstsq(in A, in b);
            for (int j = 0; j < A.N_Cols; j++)
                AssertClose(x[j], xRef[j], Band());
        }

        // ---- 1./2. dense lsqr/lsmr with a non-diagonal SPD N match the QR oracle ----
        void LsqrRightPreSpdDense()
        {
            int m = 20, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 63001);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 63002);
            var pre = new fProxyDenseSpdPreconditioner { Nmat = fProxyKrylovBatteryOracles.BuildDenseSpd(n, 63003, (fProxy)n) };

            fProxy tol = Consts.fProxySqrtEps;
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lsqrRightPre(in A, in pre, in b, ref x, 20 * n, tol);
            CheckSolution(in A, in b, in x, in info, tol);
        }

        void LsmrRightPreSpdDense()
        {
            int m = 20, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 63101);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 63102);
            var pre = new fProxyDenseSpdPreconditioner { Nmat = fProxyKrylovBatteryOracles.BuildDenseSpd(n, 63103, (fProxy)n) };

            fProxy tol = Consts.fProxySqrtEps;
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lsmrRightPre(in A, in pre, in b, ref x, 20 * n, tol);
            CheckSolution(in A, in b, in x, in info, tol);
        }

        // ---- 3./4. BSR entry points: same SPD N, same oracle ----
        void LsqrRightPreSpdBSR()
        {
            int m = 20, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 63201);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 63202);
            var bsm = DenseToBSR(in A);
            var pre = new fProxyDenseSpdPreconditioner { Nmat = fProxyKrylovBatteryOracles.BuildDenseSpd(n, 63203, (fProxy)n) };

            fProxy tol = Consts.fProxySqrtEps;
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lsqrRightPre(in bsm, in pre, in b, ref x, 20 * n, tol);
            CheckSolution(in A, in b, in x, in info, tol);
        }

        void LsmrRightPreSpdBSR()
        {
            int m = 20, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 63301);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 63302);
            var bsm = DenseToBSR(in A);
            var pre = new fProxyDenseSpdPreconditioner { Nmat = fProxyKrylovBatteryOracles.BuildDenseSpd(n, 63303, (fProxy)n) };

            fProxy tol = Consts.fProxySqrtEps;
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lsmrRightPre(in bsm, in pre, in b, ref x, 20 * n, tol);
            CheckSolution(in A, in b, in x, in info, tol);
        }

        // ---- 5. lsqrJacobi / lsmrJacobi (now the diagonal case of the RightPre path) still
        // match the QR oracle ----
        void JacobiWrappersMatchOracle()
        {
            int m = 20, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 63401);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 63402);

            fProxy tol = Consts.fProxySqrtEps;

            var xLsqr = new fProxyN(n, Allocator.Temp);
            var infoLsqr = Krylov.lsqrJacobi(in A, in b, ref xLsqr, 20 * n, tol);
            CheckSolution(in A, in b, in xLsqr, in infoLsqr, tol);

            var xLsmr = new fProxyN(n, Allocator.Temp);
            var infoLsmr = Krylov.lsmrJacobi(in A, in b, ref xLsmr, 20 * n, tol);
            CheckSolution(in A, in b, in xLsmr, in infoLsmr, tol);
        }

        // ---- 6./7. general (operator-valued) right preconditioner N = R^-1: A·N is orthonormal, so
        // lsqr/lsmr must SOLVE within a tiny iteration budget (n) even on an ill-scaled A, and still
        // land on the QR oracle. Proves both the non-symmetric wrapper AND its strength. ----
        void LsqrRightPreOpRinvDense()
        {
            int m = 24, n = 6;
            var A = BuildIllScaled(m, n, 63501);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 63502);
            var Rinv = BuildRinv(in A);

            fProxy tol = Consts.fProxySqrtEps;
            var x = new fProxyN(n, Allocator.Temp);
            // budget = n: converges only because A·R^-1 is (near-)orthonormal; CheckSolution asserts Solved.
            var info = Krylov.lsqrRightPreOp(in A, new fProxyDenseOperator(in Rinv), in b, ref x, n, tol);
            CheckSolution(in A, in b, in x, in info, tol);
        }

        void LsmrRightPreOpRinvDense()
        {
            int m = 24, n = 6;
            var A = BuildIllScaled(m, n, 63511);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 63512);
            var Rinv = BuildRinv(in A);

            fProxy tol = Consts.fProxySqrtEps;
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lsmrRightPreOp(in A, new fProxyDenseOperator(in Rinv), in b, ref x, n, tol);
            CheckSolution(in A, in b, in x, in info, tol);
        }

        // ---- 8./9. general non-symmetric N (N != N^T): the wrapped transpose must use N^T, so a
        // symmetric-only wrapper would give the WRONG answer here. Must match the QR oracle -- dense
        // and BSR entry points. ----
        void LsqrRightPreOpNonSymDense()
        {
            int m = 20, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 63601);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 63602);
            var Nmat = BuildNonSym(n, 63603);

            fProxy tol = Consts.fProxySqrtEps;
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lsqrRightPreOp(in A, new fProxyDenseOperator(in Nmat), in b, ref x, 20 * n, tol);
            CheckSolution(in A, in b, in x, in info, tol);
        }

        void LsqrRightPreOpNonSymBSR()
        {
            int m = 20, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 63701);
            var b = GenerateOP.fProxyRandomVec(m, -1f, 1f, 63702);
            var bsm = DenseToBSR(in A);
            var Nmat = BuildNonSym(n, 63703);

            fProxy tol = Consts.fProxySqrtEps;
            var x = new fProxyN(n, Allocator.Temp);
            var info = Krylov.lsqrRightPreOp(in bsm, new fProxyDenseOperator(in Nmat), in b, ref x, 20 * n, tol);
            CheckSolution(in A, in b, in x, in info, tol);
        }
    }

    // Generous timeout: the first case run in a session pays one cold Burst compile of the whole
    // Execute() body.
    [Timeout(600000)]
    [Test]
    public void LsqrRightPreSpdDense()
        => new TestJob { Type = TestJob.TestType.LsqrRightPreSpdDense }.Run();

    [Timeout(600000)]
    [Test]
    public void LsmrRightPreSpdDense()
        => new TestJob { Type = TestJob.TestType.LsmrRightPreSpdDense }.Run();

    [Timeout(600000)]
    [Test]
    public void LsqrRightPreSpdBSR()
        => new TestJob { Type = TestJob.TestType.LsqrRightPreSpdBSR }.Run();

    [Timeout(600000)]
    [Test]
    public void LsmrRightPreSpdBSR()
        => new TestJob { Type = TestJob.TestType.LsmrRightPreSpdBSR }.Run();

    [Timeout(600000)]
    [Test]
    public void JacobiWrappersMatchOracle()
        => new TestJob { Type = TestJob.TestType.JacobiWrappersMatchOracle }.Run();

    [Timeout(600000)]
    [Test]
    public void LsqrRightPreOpRinvDense()
        => new TestJob { Type = TestJob.TestType.LsqrRightPreOpRinvDense }.Run();

    [Timeout(600000)]
    [Test]
    public void LsmrRightPreOpRinvDense()
        => new TestJob { Type = TestJob.TestType.LsmrRightPreOpRinvDense }.Run();

    [Timeout(600000)]
    [Test]
    public void LsqrRightPreOpNonSymDense()
        => new TestJob { Type = TestJob.TestType.LsqrRightPreOpNonSymDense }.Run();

    [Timeout(600000)]
    [Test]
    public void LsqrRightPreOpNonSymBSR()
        => new TestJob { Type = TestJob.TestType.LsqrRightPreOpNonSymBSR }.Run();
}
