using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Block ILU(0) preconditioner (fProxyILU0) + right-preconditioned BiCGSTAB (Krylov.biCGStab).
// Anchors: (a) on a block-tridiagonal (fill-free) NONSYMMETRIC system ILU(0) is the exact LU,
// so Apply must match a dense LU solve; (b) biCGStab with ILU(0) converges to the true
// solution and needs fewer iterations than unpreconditioned biCGStab on the same system.
public class fProxySparseILU0Tests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SparseILU0TestJob : IJob
    {
        public enum TestType
        {
            ExactOnBlockTridiagonal,
            PbiCGStabConvergesAndBeatsPlain,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ExactOnBlockTridiagonal: ExactOnBlockTridiagonal(); break;
                case TestType.PbiCGStabConvergesAndBeatsPlain: PbiCGStabConvergesAndBeatsPlain(); break;
            }
        }

        // Diagonally dominant NONSYMMETRIC block-tridiagonal system (fill-free pattern).
        static fProxyBSR BuildNonsymTridiag(int nb, int BR, uint seed)
        {
            var builder = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Temp);
            var rng = new Unity.Mathematics.Random(seed);
            var blk = new fProxyMxN(BR, BR, Allocator.Temp);

            for (int i = 0; i < nb; i++)
            {
                for (int r = 0; r < BR; r++)
                    for (int c = 0; c < BR; c++)
                        blk[r, c] = (r == c ? (fProxy)(4 * BR) : (fProxy)0) + (fProxy)rng.NextFloat(-0.5f, 0.5f);
                builder.AddBlock(i, i, in blk);

                if (i + 1 < nb)
                {
                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            blk[r, c] = (fProxy)rng.NextFloat(-0.6f, 0.6f);
                    builder.AddBlock(i + 1, i, in blk);

                    for (int r = 0; r < BR; r++)
                        for (int c = 0; c < BR; c++)
                            blk[r, c] = (fProxy)rng.NextFloat(-0.6f, 0.6f);   // NOT the transpose
                    builder.AddBlock(i, i + 1, in blk);
                }
            }
            return builder.ToBSR(Allocator.Temp);
        }

        void ExactOnBlockTridiagonal()
        {
            const int nb = 6, BR = 3;
            var A = BuildNonsymTridiag(nb, BR, 841001u);
            int n = A.M_Rows;

            var M = new fProxyILU0(in A, Allocator.Temp);
            Assert.IsTrue(M.Shift == (fProxy)0);

            var r = GenerateOP.fProxyRandomVec(n, -1f, 1f, 841002u);
            var z = new fProxyN(n, Allocator.Temp);
            M.Apply(in r, ref z);

            // Dense LU oracle: on a fill-free pattern ILU(0) is exact, so z == A^-1 r.
            var D = A.ToDense(Allocator.Temp);
            var zRef = new fProxyN(n, Allocator.Temp);
            for (int i = 0; i < n; i++) zRef[i] = r[i];
            var P = new Pivot(n, Allocator.Temp);
            var info = LU.solveInPlace(ref D, ref P, ref zRef);
            Assert.IsTrue(info.Solved);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(z[i] - zRef[i]) < Tol() * ((fProxy)1 + math.abs(zRef[i])));
        }

        void PbiCGStabConvergesAndBeatsPlain()
        {
            const int nb = 40, BR = 3;
            var A = BuildNonsymTridiag(nb, BR, 841003u);
            int n = A.M_Rows;

            var xTrue = GenerateOP.fProxyRandomVec(n, 0.5f, 1.5f, 841004u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 4 * n;

            // Plain BiCGSTAB baseline.
            var xP = new fProxyN(n, Allocator.Temp);
            var rP = new fProxyN(n, Allocator.Temp); var rh = new fProxyN(n, Allocator.Temp); var pP = new fProxyN(n, Allocator.Temp);
            var vP = new fProxyN(n, Allocator.Temp); var tP = new fProxyN(n, Allocator.Temp);
            var op = new fProxyBSROperator(in A);
            var infoPlain = Krylov.biCGStab(in op, in b, ref xP, ref rP, ref rh, ref pP, ref vP, ref tP, maxIter, tol);
            Assert.IsTrue(infoPlain.Solved);

            // ILU(0)-preconditioned.
            var M = new fProxyILU0(in A, Allocator.Temp);
            var xI = new fProxyN(n, Allocator.Temp);
            var infoIlu = Krylov.biCGStab(in A, in M, in b, ref xI, maxIter, tol);
            Assert.IsTrue(infoIlu.Solved);
            Assert.IsTrue((double)infoIlu.iterations <= (double)infoPlain.iterations * 0.9);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(xI[i] - xTrue[i]) < Tol() * ((fProxy)1 + math.abs(xTrue[i])));
        }
    }

    [Test]
    public void ExactOnBlockTridiagonalTest()
        => new SparseILU0TestJob { Type = SparseILU0TestJob.TestType.ExactOnBlockTridiagonal }.Run();

    [Test]
    public void PbiCGStabConvergesAndBeatsPlainTest()
        => new SparseILU0TestJob { Type = SparseILU0TestJob.TestType.PbiCGStabConvergesAndBeatsPlain }.Run();

    [Test]
    public void MissingDiagonalThrows()
    {
        var builder = new fProxyBSRBuilder(2, 2, 2, 2, Allocator.Temp);
        var block = GenerateOP.fProxyMat(2, 2, (fProxy)1);
        builder.AddBlock(0, 0, in block);
        builder.AddBlock(1, 0, in block);
        var A = builder.ToBSR(Allocator.Temp);
        Assert.Throws<ArgumentException>(() => { var m = new fProxyILU0(in A, Allocator.Temp); });
    }
}
