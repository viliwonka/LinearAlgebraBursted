using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// AᵀA-Jacobi (column-equilibration) least-squares preconditioner primitives:
//   Blas.columnNormsSquared (dense) / BSR.columnNormsSquared (BSR),
//   Blas.buildJacobiScale, and floatColScaledOperator<TInner> (the A·D wrapper).
// Value cases run inside a [BurstCompile] IJob (matches the other sparse-solver suites);
// the Symmetric-BSR reject runs on the managed thread with Assert.Throws.
public class floatJacobiPrecondTests
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

        static float Tight() => 1e-4f;
        static float LooseTol() => 1e-2f;

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
        static floatBSR DenseToBSR(ref Arena arena, in floatMxN A, int BR, int BC, int nnzHint)
        {
            var builder = arena.floatBSRBuilder(A.M_Rows / BR, A.N_Cols / BC, BR, BC, math.max(nnzHint, 1));
            for (int r = 0; r < A.M_Rows; r++)
                for (int c = 0; c < A.N_Cols; c++)
                    if (A[r, c] != (float)0)
                        builder.AddValue(r, c, A[r, c]);
            return builder.ToBSR(ref arena);
        }

        static void AssertClose(float got, float expected, float tol)
            => Assert.IsTrue(math.abs(got - expected) <= tol * ((float)1 + math.abs(expected)));

        // ---- 1. dense columnNormsSquared == diag(AᵀA) ----
        void ColumnNormsSquaredDenseMatchesReference()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 9, n = 6;
            var A = arena.floatRandomMat(m, n, -1f, 1f, 61001);

            var d2 = arena.floatVec(n);
            Blas.columnNormsSquared(in A, ref d2);

            for (int c = 0; c < n; c++)
            {
                float refv = (float)0;
                for (int r = 0; r < m; r++) refv += A[r, c] * A[r, c];
                AssertClose(d2[c], refv, Tight());
            }

            arena.Dispose();
        }

        // ---- 2. BSR columnNormsSquared (1x1 and 3x3 blocks) matches dense ----
        void ColumnNormsSquaredBSRMatchesDense()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 9, n = 6;                                 // multiples of 3 for the 3x3 case
            var A = arena.floatRandomMat(m, n, -1f, 1f, 61101);

            var d2Dense = arena.floatVec(n);
            Blas.columnNormsSquared(in A, ref d2Dense);

            var bsm1 = DenseToBSR(ref arena, in A, 1, 1, m * n);
            var d2b1 = arena.floatVec(n);
            BSR.columnNormsSquared(in bsm1, ref d2b1);
            for (int c = 0; c < n; c++) AssertClose(d2b1[c], d2Dense[c], Tight());

            var bsm3 = DenseToBSR(ref arena, in A, 3, 3, m * n);
            var d2b3 = arena.floatVec(n);
            BSR.columnNormsSquared(in bsm3, ref d2b3);
            for (int c = 0; c < n; c++) AssertClose(d2b3[c], d2Dense[c], Tight());

            // RECTANGULAR blocks BR=2, BC=3 (would catch a BR/BC swap in the block-column indexing
            // that square blocks cannot). Fresh matrix sized to the block grid: 4x9 -> 2x3 grid.
            int m2 = 4, n2 = 9;
            var A2 = arena.floatRandomMat(m2, n2, -1f, 1f, 61102);
            var d2Dense2 = arena.floatVec(n2);
            Blas.columnNormsSquared(in A2, ref d2Dense2);

            var bsm23 = DenseToBSR(ref arena, in A2, 2, 3, m2 * n2);
            var d2b23 = arena.floatVec(n2);
            BSR.columnNormsSquared(in bsm23, ref d2b23);
            for (int c = 0; c < n2; c++) AssertClose(d2b23[c], d2Dense2[c], Tight());

            arena.Dispose();
        }

        // ---- 3. buildJacobiScale: d = 1/sqrt(colNorm2), zero column -> d = 1 ----
        void BuildJacobiScaleZeroColumnGuard()
        {
            var arena = new Arena(Allocator.Persistent);

            var c2 = arena.floatVec(4);
            c2[0] = (float)4;    c2[1] = (float)0;    c2[2] = (float)9;   c2[3] = (float)0.25;
            var d = arena.floatVec(4);
            Blas.buildJacobiScale(in c2, ref d);

            AssertClose(d[0], (float)0.5, Tight());        // 1/sqrt(4)
            AssertClose(d[1], (float)1,   Tight());        // zero column -> unscaled
            AssertClose(d[2], (float)1 / (float)3, Tight()); // 1/sqrt(9)
            AssertClose(d[3], (float)2,   Tight());        // 1/sqrt(0.25)

            for (int j = 0; j < 4; j++) Assert.IsTrue(math.isfinite(d[j]));

            arena.Dispose();
        }

        // ---- 4. adjoint identity <(AD)v, u> == <v, (AD)^T u> ----
        void ColScaledOperatorAdjointIdentity()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 9, n = 6;
            var A = arena.floatRandomMat(m, n, -1f, 1f, 61301);
            var d2 = arena.floatVec(n);
            Blas.columnNormsSquared(in A, ref d2);
            var d = arena.floatVec(n);
            Blas.buildJacobiScale(in d2, ref d);
            var scratch = arena.floatVec(n);

            var op = new floatColScaledOperator<floatDenseOperator>(new floatDenseOperator(in A), d, scratch);

            var u = arena.floatRandomVec(m, -1f, 1f, 61302);
            var v = arena.floatRandomVec(n, -1f, 1f, 61303);

            var y1 = arena.floatVec(m);
            op.Apply(in v, ref y1);                 // (AD) v
            float lhs = Blas.dot(y1, u);

            var y2 = arena.floatVec(n);
            op.ApplyT(in u, ref y2);                // (AD)^T u
            float rhs = Blas.dot(v, y2);

            AssertClose(lhs, rhs, LooseTol());

            arena.Dispose();
        }

        // ---- 5. equilibration: columns of A*D have unit norm ----
        void ColScaledEquilibrationUnitColumns()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 12, n = 5;
            var A = arena.floatRandomMat(m, n, -1f, 1f, 61401);
            var d2 = arena.floatVec(n);
            Blas.columnNormsSquared(in A, ref d2);
            var d = arena.floatVec(n);
            Blas.buildJacobiScale(in d2, ref d);

            // Materialize A*D (scale column j by d[j]) and re-measure its column norms.
            var AD = arena.floatMat(m, n);
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++)
                    AD[r, c] = A[r, c] * d[c];

            var d2s = arena.floatVec(n);
            Blas.columnNormsSquared(in AD, ref d2s);
            for (int c = 0; c < n; c++) AssertClose(d2s[c], (float)1, LooseTol());  // random cols are nonzero

            arena.Dispose();
        }

        // ---- 6. composability: preconditioned lsqr solves a consistent over-determined system ----
        // Solve (A D) y = b with the wrapped operator, recover x = D y; x must equal x_true and A x == b.
        void ColScaledOperatorSolvesConsistent()
        {
            var arena = new Arena(Allocator.Persistent);

            int m = 10, n = 4;
            var A = arena.floatRandomMat(m, n, -1f, 1f, 61501);
            var xTrue = arena.floatRandomVec(n, -1f, 1f, 61502);
            var b = Blas.dot(A, xTrue);        // consistent

            var d2 = arena.floatVec(n);
            Blas.columnNormsSquared(in A, ref d2);
            var d = arena.floatVec(n);
            Blas.buildJacobiScale(in d2, ref d);
            var scratch = arena.floatVec(n);

            var op = new floatColScaledOperator<floatDenseOperator>(new floatDenseOperator(in A), d, scratch);

            var y    = arena.floatVec(n);
            var u    = arena.floatVec(m);
            var vv   = arena.floatVec(n);
            var w    = arena.floatVec(n);
            var tmpM = arena.floatVec(m);
            var tmpN = arena.floatVec(n);
            bool ok = Krylov.lsqr(op, in b, ref y, ref u, ref vv, ref w, ref tmpM, ref tmpN, 8 * n, Consts.floatSqrtEps);
            Assert.IsTrue(ok);

            var x = arena.floatVec(n);
            for (int j = 0; j < n; j++) x[j] = d[j] * y[j];    // unscale

            for (int j = 0; j < n; j++) AssertClose(x[j], xTrue[j], LooseTol());

            var Ax = Blas.dot(A, x);
            for (int i = 0; i < m; i++) AssertClose(Ax[i], b[i], LooseTol());

            arena.Dispose();
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
        var arena = new Arena(Allocator.Persistent);

        // A diagonal 1x1-block matrix is trivially symmetric -> ToBSRSymmetric accepts it.
        var s = arena.floatBSRBuilder(3, 3, 1, 1, 3);
        s.AddValue(0, 0, (float)2);
        s.AddValue(1, 1, (float)3);
        s.AddValue(2, 2, (float)4);
        var sym = s.ToBSRSymmetric(ref arena);

        var d2 = arena.floatVec(3);
        Assert.Throws<ArgumentException>(() => BSR.columnNormsSquared(in sym, ref d2));

        arena.Dispose();
    }
}
