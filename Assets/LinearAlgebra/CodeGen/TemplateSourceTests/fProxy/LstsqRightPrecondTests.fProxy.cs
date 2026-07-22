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
    // Dense SPD right preconditioner: z = Nmat * r via a plain dense mat-vec. Nmat must be
    // symmetric positive definite (the tests build it as I + W^T W / n).
    public struct SpdPre : IfProxyPreconditioner
    {
        public fProxyMxN Nmat;   // n x n symmetric positive definite

        public bool IsIdentity => false;

        public void Apply(in fProxyN r, ref fProxyN z) => Blas.dot(in Nmat, in r, ref z);
    }

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

        // Non-diagonal SPD preconditioner matrix N = I + W^T W / n (W random n x n). Bit-exactly
        // symmetric (the (i,j) and (j,i) sums run the same k order), eigenvalues >= 1, mild
        // condition number -- a well-conditioned but genuinely non-diagonal symmetric N.
        static fProxyMxN BuildSpd(ref Arena arena, int n, uint seed)
        {
            var W = arena.fProxyRandomMat(n, n, -1f, 1f, seed);
            var Nmat = arena.fProxyMat(n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    fProxy s = (fProxy)0;
                    for (int k = 0; k < n; k++) s += W[k, i] * W[k, j];
                    Nmat[i, j] = s / (fProxy)n + (i == j ? (fProxy)1 : (fProxy)0);
                }
            return Nmat;
        }

        // R^-1 (n x n dense) from the thin QR of A (A full column rank): A·R^-1 = Q has orthonormal
        // columns, so it is the canonical STRONG least-squares right preconditioner (Blendenpik/LSRN
        // shape) -- and non-symmetric (upper-triangular). Built by upper-triangular back-substitution
        // solving R·Rinv = I column by column.
        static fProxyMxN BuildRinv(ref Arena arena, in fProxyMxN A)
        {
            int n = A.N_Cols;
            var Q = arena.fProxyMat(A.M_Rows, n);
            var R = arena.fProxyMat(n);
            QR.decomp(in A, ref Q, ref R);
            var Rinv = arena.fProxyMat(n);
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
        static fProxyMxN BuildNonSym(ref Arena arena, int n, uint seed)
        {
            var W = arena.fProxyRandomMat(n, n, -1f, 1f, seed);
            var Nmat = arena.fProxyMat(n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    Nmat[i, j] = (i == j) ? (fProxy)1 : (j < i ? (fProxy)0.25 * W[i, j] : (fProxy)0);
            return Nmat;
        }

        // Random m x n A with columns geometrically scaled to condition number ~1e2 -- ill-scaled
        // enough that a column preconditioner earns its keep, mild enough to stay inside the float band.
        static fProxyMxN BuildIllScaled(ref Arena arena, int m, int n, uint seed)
        {
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, seed);
            for (int j = 0; j < n; j++)
            {
                fProxy scale = math.pow((fProxy)10, (fProxy)2 * j / (fProxy)(n - 1));
                for (int i = 0; i < m; i++) A[i, j] *= scale;
            }
            return A;
        }

        // Direct dense least-squares reference via QR (A full column rank, M_Rows >= N_Cols).
        static fProxyN ReferenceSolveLstsq(ref Arena arena, in fProxyMxN A, in fProxyN b)
        {
            var Q = arena.fProxyMat(A.M_Rows, A.N_Cols);
            var R = arena.fProxyMat(A.N_Cols);
            QR.decomp(in A, ref Q, ref R);
            fProxyN bLocal = b;
            var xRef = arena.fProxyVec(A.N_Cols);
            QR.decompSolve(ref Q, ref R, ref bLocal, ref xRef);
            return xRef;
        }

        // 1x1-block BSR copy of a dense matrix (scalar AddValue per nonzero).
        static fProxyBSR DenseToBSR(ref Arena arena, in fProxyMxN A)
        {
            var builder = arena.fProxyBSRBuilder(A.M_Rows, A.N_Cols, 1, 1, A.M_Rows * A.N_Cols);
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSR(ref arena);
        }

        static void AssertClose(fProxy got, fProxy expected, fProxy tol)
            => Assert.IsTrue(math.abs(got - expected) <= tol * ((fProxy)1 + math.abs(expected)));

        // Shared checks on a solved system: converged, exact recomputed ||A^T r|| small on the
        // fixed ||A^T b|| scale (100x headroom over tol covers cond(N) mapping the y-space
        // stopping test back to original coordinates), elementwise agreement with the QR oracle.
        static void CheckSolution(ref Arena arena, in fProxyMxN A, in fProxyN b, in fProxyN x, in LstsqInfo info, fProxy tol)
        {
            Assert.IsTrue(info.Solved);

            var atb = arena.fProxyVec(A.N_Cols);
            Blas.dot(in b, in A, ref atb);                    // A^T b
            fProxy atbNorm = math.sqrt(Blas.dot(atb, atb));
            Assert.IsTrue((fProxy)info.Arnorm <= (fProxy)100 * tol * math.max(atbNorm, (fProxy)1e-30));

            var xRef = ReferenceSolveLstsq(ref arena, in A, in b);
            for (int j = 0; j < A.N_Cols; j++)
                AssertClose(x[j], xRef[j], Band());
        }

        // ---- 1./2. dense lsqr/lsmr with a non-diagonal SPD N match the QR oracle ----
        void LsqrRightPreSpdDense()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 20, n = 6;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 63001);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 63002);
            var pre = new SpdPre { Nmat = BuildSpd(ref arena, n, 63003) };

            fProxy tol = Consts.fProxySqrtEps;
            var x = arena.fProxyVec(n);
            var info = Krylov.lsqrRightPre(in A, in pre, in b, ref x, 20 * n, tol);
            CheckSolution(ref arena, in A, in b, in x, in info, tol);

            arena.Dispose();
        }

        void LsmrRightPreSpdDense()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 20, n = 6;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 63101);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 63102);
            var pre = new SpdPre { Nmat = BuildSpd(ref arena, n, 63103) };

            fProxy tol = Consts.fProxySqrtEps;
            var x = arena.fProxyVec(n);
            var info = Krylov.lsmrRightPre(in A, in pre, in b, ref x, 20 * n, tol);
            CheckSolution(ref arena, in A, in b, in x, in info, tol);

            arena.Dispose();
        }

        // ---- 3./4. BSR entry points: same SPD N, same oracle ----
        void LsqrRightPreSpdBSR()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 20, n = 6;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 63201);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 63202);
            var bsm = DenseToBSR(ref arena, in A);
            var pre = new SpdPre { Nmat = BuildSpd(ref arena, n, 63203) };

            fProxy tol = Consts.fProxySqrtEps;
            var x = arena.fProxyVec(n);
            var info = Krylov.lsqrRightPre(in bsm, in pre, in b, ref x, 20 * n, tol);
            CheckSolution(ref arena, in A, in b, in x, in info, tol);

            arena.Dispose();
        }

        void LsmrRightPreSpdBSR()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 20, n = 6;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 63301);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 63302);
            var bsm = DenseToBSR(ref arena, in A);
            var pre = new SpdPre { Nmat = BuildSpd(ref arena, n, 63303) };

            fProxy tol = Consts.fProxySqrtEps;
            var x = arena.fProxyVec(n);
            var info = Krylov.lsmrRightPre(in bsm, in pre, in b, ref x, 20 * n, tol);
            CheckSolution(ref arena, in A, in b, in x, in info, tol);

            arena.Dispose();
        }

        // ---- 5. lsqrJacobi / lsmrJacobi (now the diagonal case of the RightPre path) still
        // match the QR oracle ----
        void JacobiWrappersMatchOracle()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 20, n = 6;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 63401);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 63402);

            fProxy tol = Consts.fProxySqrtEps;

            var xLsqr = arena.fProxyVec(n);
            var infoLsqr = Krylov.lsqrJacobi(in A, in b, ref xLsqr, 20 * n, tol);
            CheckSolution(ref arena, in A, in b, in xLsqr, in infoLsqr, tol);

            var xLsmr = arena.fProxyVec(n);
            var infoLsmr = Krylov.lsmrJacobi(in A, in b, ref xLsmr, 20 * n, tol);
            CheckSolution(ref arena, in A, in b, in xLsmr, in infoLsmr, tol);

            arena.Dispose();
        }

        // ---- 6./7. general (operator-valued) right preconditioner N = R^-1: A·N is orthonormal, so
        // lsqr/lsmr must SOLVE within a tiny iteration budget (n) even on an ill-scaled A, and still
        // land on the QR oracle. Proves both the non-symmetric wrapper AND its strength. ----
        void LsqrRightPreOpRinvDense()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 24, n = 6;
            var A = BuildIllScaled(ref arena, m, n, 63501);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 63502);
            var Rinv = BuildRinv(ref arena, in A);

            fProxy tol = Consts.fProxySqrtEps;
            var x = arena.fProxyVec(n);
            // budget = n: converges only because A·R^-1 is (near-)orthonormal; CheckSolution asserts Solved.
            var info = Krylov.lsqrRightPreOp(in A, new fProxyDenseOperator(in Rinv), in b, ref x, n, tol);
            CheckSolution(ref arena, in A, in b, in x, in info, tol);

            arena.Dispose();
        }

        void LsmrRightPreOpRinvDense()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 24, n = 6;
            var A = BuildIllScaled(ref arena, m, n, 63511);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 63512);
            var Rinv = BuildRinv(ref arena, in A);

            fProxy tol = Consts.fProxySqrtEps;
            var x = arena.fProxyVec(n);
            var info = Krylov.lsmrRightPreOp(in A, new fProxyDenseOperator(in Rinv), in b, ref x, n, tol);
            CheckSolution(ref arena, in A, in b, in x, in info, tol);

            arena.Dispose();
        }

        // ---- 8./9. general non-symmetric N (N != N^T): the wrapped transpose must use N^T, so a
        // symmetric-only wrapper would give the WRONG answer here. Must match the QR oracle -- dense
        // and BSR entry points. ----
        void LsqrRightPreOpNonSymDense()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 20, n = 6;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 63601);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 63602);
            var Nmat = BuildNonSym(ref arena, n, 63603);

            fProxy tol = Consts.fProxySqrtEps;
            var x = arena.fProxyVec(n);
            var info = Krylov.lsqrRightPreOp(in A, new fProxyDenseOperator(in Nmat), in b, ref x, 20 * n, tol);
            CheckSolution(ref arena, in A, in b, in x, in info, tol);

            arena.Dispose();
        }

        void LsqrRightPreOpNonSymBSR()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 20, n = 6;
            var A = arena.fProxyRandomMat(m, n, -1f, 1f, 63701);
            var b = arena.fProxyRandomVec(m, -1f, 1f, 63702);
            var bsm = DenseToBSR(ref arena, in A);
            var Nmat = BuildNonSym(ref arena, n, 63703);

            fProxy tol = Consts.fProxySqrtEps;
            var x = arena.fProxyVec(n);
            var info = Krylov.lsqrRightPreOp(in bsm, new fProxyDenseOperator(in Nmat), in b, ref x, 20 * n, tol);
            CheckSolution(ref arena, in A, in b, in x, in info, tol);

            arena.Dispose();
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
