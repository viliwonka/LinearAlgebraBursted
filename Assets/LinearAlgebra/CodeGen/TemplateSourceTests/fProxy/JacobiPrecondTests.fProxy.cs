using System;
using BULA;
using BULA.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// AᵀA-Jacobi (column-equilibration) least-squares preconditioner primitives:
//   Blas.columnNormsSquared (dense) / BSR.columnNormsSquared (BSR),
//   Blas.buildJacobiScale, and fProxyColScaledOperator<TInner> (the A·D wrapper).
// Value cases run inside a [BurstCompile] IJob (matches the other sparse-solver suites);
// the Symmetric-BSR reject runs on the managed thread with Assert.Throws.
public class fProxyJacobiPrecondTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct JacobiPrecondTestJob : IJob
    {
        public enum TestType
        {
            ColumnNormsSquaredDenseMatchesReference,
            ColumnNormsSquaredBSRMatchesDense,
            BuildJacobiScaleZeroColumnGuard,
            ColScaledOperatorAdjointIdentity,
            ColScaledEquilibrationUnitColumns,
            ColScaledOperatorSolvesConsistent,
        }

        public TestType Type;

        static fProxy Tight() => /*+choose[1e-4f|1e-11]*/1e-4f/*-choose*/;
        static fProxy LooseTol() => /*+choose[1e-2f|1e-5]*/1e-2f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ColumnNormsSquaredDenseMatchesReference: ColumnNormsSquaredDenseMatchesReference(); break;
                case TestType.ColumnNormsSquaredBSRMatchesDense: ColumnNormsSquaredBSRMatchesDense(); break;
                case TestType.BuildJacobiScaleZeroColumnGuard: BuildJacobiScaleZeroColumnGuard(); break;
                case TestType.ColScaledOperatorAdjointIdentity: ColScaledOperatorAdjointIdentity(); break;
                case TestType.ColScaledEquilibrationUnitColumns: ColScaledEquilibrationUnitColumns(); break;
                case TestType.ColScaledOperatorSolvesConsistent: ColScaledOperatorSolvesConsistent(); break;
            }
        }

        // ---- helpers ----

        // BR x BC-block BSR built from a dense matrix's nonzero entries via scalar AddValue.
        static fProxyBSR DenseToBSR(in fProxyMxN A, int BR, int BC, int nnzHint)
        {
            var builder = new fProxyBSRBuilder(A.M_Rows / BR, A.N_Cols / BC, BR, BC, Allocator.Temp, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (fProxy)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSR(Allocator.Temp);
        }

        // ---- 1. dense columnNormsSquared == diag(AᵀA) ----
        void ColumnNormsSquaredDenseMatchesReference()
        {
            int m = 9, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 61001);

            var d2 = new fProxyN(n, Allocator.Temp);
            Blas.columnNormsSquared(in A, ref d2);

            for (int c = 0; c < n; c++)
            {
                fProxy refv = (fProxy)0;
                for (int r = 0; r < m; r++) refv += A[r, c] * A[r, c];
                fProxyKrylovTestAsserts.AssertClose(d2[c], refv, Tight());
            }
        }

        // ---- 2. BSR columnNormsSquared (1x1 and 3x3 blocks) matches dense ----
        void ColumnNormsSquaredBSRMatchesDense()
        {
            int m = 9, n = 6;                                 // multiples of 3 for the 3x3 case
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 61101);

            var d2Dense = new fProxyN(n, Allocator.Temp);
            Blas.columnNormsSquared(in A, ref d2Dense);

            var bsm1 = DenseToBSR(in A, 1, 1, m * n);
            var d2b1 = new fProxyN(n, Allocator.Temp);
            BSR.columnNormsSquared(in bsm1, ref d2b1);
            for (int c = 0; c < n; c++) fProxyKrylovTestAsserts.AssertClose(d2b1[c], d2Dense[c], Tight());

            var bsm3 = DenseToBSR(in A, 3, 3, m * n);
            var d2b3 = new fProxyN(n, Allocator.Temp);
            BSR.columnNormsSquared(in bsm3, ref d2b3);
            for (int c = 0; c < n; c++) fProxyKrylovTestAsserts.AssertClose(d2b3[c], d2Dense[c], Tight());

            // RECTANGULAR blocks BR=2, BC=3 (would catch a BR/BC swap in the block-column indexing
            // that square blocks cannot). Fresh matrix sized to the block grid: 4x9 -> 2x3 grid.
            int m2 = 4, n2 = 9;
            var A2 = GenerateOP.fProxyRandomMat(m2, n2, -1f, 1f, 61102);
            var d2Dense2 = new fProxyN(n2, Allocator.Temp);
            Blas.columnNormsSquared(in A2, ref d2Dense2);

            var bsm23 = DenseToBSR(in A2, 2, 3, m2 * n2);
            var d2b23 = new fProxyN(n2, Allocator.Temp);
            BSR.columnNormsSquared(in bsm23, ref d2b23);
            for (int c = 0; c < n2; c++) fProxyKrylovTestAsserts.AssertClose(d2b23[c], d2Dense2[c], Tight());
        }

        // ---- 3. buildJacobiScale: d = 1/sqrt(colNorm2), zero column -> d = 1 ----
        void BuildJacobiScaleZeroColumnGuard()
        {
            var c2 = new fProxyN(4, Allocator.Temp);
            c2[0] = (fProxy)4;    c2[1] = (fProxy)0;    c2[2] = (fProxy)9;   c2[3] = (fProxy)0.25;
            var d = new fProxyN(4, Allocator.Temp);
            Blas.buildJacobiScale(in c2, ref d);

            fProxyKrylovTestAsserts.AssertClose(d[0], (fProxy)0.5, Tight());        // 1/sqrt(4)
            fProxyKrylovTestAsserts.AssertClose(d[1], (fProxy)1,   Tight());        // zero column -> unscaled
            fProxyKrylovTestAsserts.AssertClose(d[2], (fProxy)1 / (fProxy)3, Tight()); // 1/sqrt(9)
            fProxyKrylovTestAsserts.AssertClose(d[3], (fProxy)2,   Tight());        // 1/sqrt(0.25)

            for (int j = 0; j < 4; j++) Assert.IsTrue(math.isfinite(d[j]));
        }

        // ---- 4. adjoint identity <(AD)v, u> == <v, (AD)^T u> ----
        void ColScaledOperatorAdjointIdentity()
        {
            int m = 9, n = 6;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 61301);
            var d2 = new fProxyN(n, Allocator.Temp);
            Blas.columnNormsSquared(in A, ref d2);
            var d = new fProxyN(n, Allocator.Temp);
            Blas.buildJacobiScale(in d2, ref d);
            var scratch = new fProxyN(n, Allocator.Temp);

            var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);

            var u = GenerateOP.fProxyRandomVec(m, -1f, 1f, 61302);
            var v = GenerateOP.fProxyRandomVec(n, -1f, 1f, 61303);

            var y1 = new fProxyN(m, Allocator.Temp);
            op.Apply(in v, ref y1);                 // (AD) v
            fProxy lhs = Blas.dot(y1, u);

            var y2 = new fProxyN(n, Allocator.Temp);
            op.ApplyT(in u, ref y2);                // (AD)^T u
            fProxy rhs = Blas.dot(v, y2);

            fProxyKrylovTestAsserts.AssertClose(lhs, rhs, LooseTol());
        }

        // ---- 5. equilibration: columns of A*D have unit norm ----
        void ColScaledEquilibrationUnitColumns()
        {
            int m = 12, n = 5;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 61401);
            var d2 = new fProxyN(n, Allocator.Temp);
            Blas.columnNormsSquared(in A, ref d2);
            var d = new fProxyN(n, Allocator.Temp);
            Blas.buildJacobiScale(in d2, ref d);

            // Materialize A*D (scale column j by d[j]) and re-measure its column norms.
            var AD = new fProxyMxN(m, n, Allocator.Temp);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    AD[r, c] = A[r, c] * d[c];

            var d2s = new fProxyN(n, Allocator.Temp);
            Blas.columnNormsSquared(in AD, ref d2s);
            for (int c = 0; c < n; c++) fProxyKrylovTestAsserts.AssertClose(d2s[c], (fProxy)1, LooseTol());  // random cols are nonzero
        }

        // ---- 6. composability: preconditioned lsqr solves a consistent over-determined system ----
        // Solve (A D) y = b with the wrapped operator, recover x = D y; x must equal x_true and A x == b.
        void ColScaledOperatorSolvesConsistent()
        {
            int m = 10, n = 4;
            var A = GenerateOP.fProxyRandomMat(m, n, -1f, 1f, 61501);
            var xTrue = GenerateOP.fProxyRandomVec(n, -1f, 1f, 61502);
            var b = Blas.dot(A, xTrue);        // consistent

            var d2 = new fProxyN(n, Allocator.Temp);
            Blas.columnNormsSquared(in A, ref d2);
            var d = new fProxyN(n, Allocator.Temp);
            Blas.buildJacobiScale(in d2, ref d);
            var scratch = new fProxyN(n, Allocator.Temp);

            var op = new fProxyColScaledOperator<fProxyDenseOperator>(new fProxyDenseOperator(in A), d, scratch);

            var y    = new fProxyN(n, Allocator.Temp);
            var u    = new fProxyN(m, Allocator.Temp);
            var vv   = new fProxyN(n, Allocator.Temp);
            var w    = new fProxyN(n, Allocator.Temp);
            var tmpM = new fProxyN(m, Allocator.Temp);
            var tmpN = new fProxyN(n, Allocator.Temp);
            bool ok = Krylov.lsqr(op, in b, ref y, ref u, ref vv, ref w, ref tmpM, ref tmpN, 8 * n, Consts.fProxySqrtEps);
            Assert.IsTrue(ok);

            var x = new fProxyN(n, Allocator.Temp);
            for (int j = 0; j < n; j++) x[j] = d[j] * y[j];    // unscale

            for (int j = 0; j < n; j++) fProxyKrylovTestAsserts.AssertClose(x[j], xTrue[j], LooseTol());

            var Ax = Blas.dot(A, x);
            for (int i = 0; i < m; i++) fProxyKrylovTestAsserts.AssertClose(Ax[i], b[i], LooseTol());
        }
    }

    [Test]
    public void ColumnNormsSquaredDenseMatchesReferenceTest()
        => new JacobiPrecondTestJob { Type = JacobiPrecondTestJob.TestType.ColumnNormsSquaredDenseMatchesReference }.Run();

    [Test]
    public void ColumnNormsSquaredBSRMatchesDenseTest()
        => new JacobiPrecondTestJob { Type = JacobiPrecondTestJob.TestType.ColumnNormsSquaredBSRMatchesDense }.Run();

    [Test]
    public void BuildJacobiScaleZeroColumnGuardTest()
        => new JacobiPrecondTestJob { Type = JacobiPrecondTestJob.TestType.BuildJacobiScaleZeroColumnGuard }.Run();

    [Test]
    public void ColScaledOperatorAdjointIdentityTest()
        => new JacobiPrecondTestJob { Type = JacobiPrecondTestJob.TestType.ColScaledOperatorAdjointIdentity }.Run();

    [Test]
    public void ColScaledEquilibrationUnitColumnsTest()
        => new JacobiPrecondTestJob { Type = JacobiPrecondTestJob.TestType.ColScaledEquilibrationUnitColumns }.Run();

    [Test]
    public void ColScaledOperatorSolvesConsistentTest()
        => new JacobiPrecondTestJob { Type = JacobiPrecondTestJob.TestType.ColScaledOperatorSolvesConsistent }.Run();

    // ---- managed-thread reject: Symmetric BSR is not supported by columnNormsSquared ----
    [Test]
    public void ColumnNormsSquaredBSRSymmetricThrows()
    {
        // A diagonal 1x1-block matrix is trivially symmetric -> ToBSRSymmetric accepts it.
        var s = new fProxyBSRBuilder(3, 3, 1, 1, Allocator.Temp, 3);
        s.AddValue(0, 0, (fProxy)2);
        s.AddValue(1, 1, (fProxy)3);
        s.AddValue(2, 2, (fProxy)4);
        var sym = s.ToBSRSymmetric(Allocator.Temp);

        var d2 = new fProxyN(3, Allocator.Temp);
        Assert.Throws<ArgumentException>(() => BSR.columnNormsSquared(in sym, ref d2));
    }
}
