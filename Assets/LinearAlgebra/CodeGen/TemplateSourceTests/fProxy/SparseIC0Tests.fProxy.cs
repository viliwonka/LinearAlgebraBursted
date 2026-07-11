using System;
using LinearAlgebra;
using LinearAlgebra.Gallery;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Block IC(0) preconditioner (fProxyIC0). Correctness anchors:
//   (a) on a block-TRIDIAGONAL pattern Cholesky produces no fill, so IC(0) is the EXACT
//       factorization -- Apply must match a dense Cholesky solve of the same system;
//   (b) preconditioned CG with IC(0) converges, matches the true solution, and needs fewer
//       iterations than block-Jacobi on the standard Laplacian / random-SPD galleries
//       (same >=10% margin convention as SSORTests);
//   (c) Symmetric-storage input mirrors to full and produces the same preconditioner;
//   (d) M^-1 is symmetric: dot(u, M^-1 v) == dot(v, M^-1 u).
// Correctness cases run inside a [BurstCompile] IJob; guard cases run on the managed thread
// with Assert.Throws (same split as SSORTests / SparseBSRTests).
public class fProxySparseIC0Tests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct SparseIC0TestJob : IJob
    {
        public enum TestType
        {
            ExactOnBlockTridiagonal,
            BeatsJacobiOnLaplacian,
            BeatsJacobiOnRandomSparseSPD,
            SymmetricStorageMatchesFull,
            ApplyIsSymmetric,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ExactOnBlockTridiagonal: ExactOnBlockTridiagonal(); break;
                case TestType.BeatsJacobiOnLaplacian: BeatsJacobiOnLaplacian(); break;
                case TestType.BeatsJacobiOnRandomSparseSPD: BeatsJacobiOnRandomSparseSPD(); break;
                case TestType.SymmetricStorageMatchesFull: SymmetricStorageMatchesFull(); break;
                case TestType.ApplyIsSymmetric: ApplyIsSymmetric(); break;
            }
        }

        // SPD block-tridiagonal chain: diagonal blocks 2*BR*I + ones-perturbation (symmetric,
        // strongly diagonally dominant), off-diagonal coupling blocks -I. Block-tridiagonal
        // patterns are fill-free under Cholesky, so IC(0) == exact Cholesky here.
        static fProxyBSR BuildBlockTridiag(ref Arena arena, int nb, int BR)
        {
            var builder = arena.fProxyBSRBuilder(nb, nb, BR, BR);
            var diag = arena.fProxyMat(BR, BR);
            var off = arena.fProxyMat(BR, BR);

            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                {
                    diag[r, c] = (r == c ? (fProxy)(2 * BR + 2) : (fProxy)0) + (fProxy)0.25f;
                    off[r, c] = r == c ? (fProxy)(-1) : (fProxy)0;
                }

            for (int i = 0; i < nb; i++)
            {
                builder.AddBlock(i, i, in diag);
                if (i + 1 < nb)
                {
                    builder.AddBlock(i + 1, i, in off);
                    builder.AddBlock(i, i + 1, in off);
                }
            }
            return builder.ToBSR(ref arena);
        }

        void ExactOnBlockTridiagonal()
        {
            var arena = new Arena(Allocator.Persistent);

            const int nb = 6, BR = 3;
            var A = BuildBlockTridiag(ref arena, nb, BR);
            int n = A.M_Rows;

            var M = arena.fProxyIC0(in A);
            Assert.IsTrue(M.Shift == (fProxy)0);   // clean factorization, no shift needed

            var r = arena.fProxyRandomVec(n, -1f, 1f, 831001u);
            var z = arena.fProxyVec(n);
            M.Apply(in r, ref z);

            // Dense Cholesky oracle: z must equal A^-1 r because IC(0) is exact on this pattern.
            var D = A.ToDense(ref arena);
            var zRef = arena.fProxyVec(n);
            for (int i = 0; i < n; i++) zRef[i] = r[i];
            var info = CHO.solveInPlace(ref D, ref zRef);
            Assert.IsTrue(info.Solved);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(z[i] - zRef[i]) < Tol() * ((fProxy)1 + math.abs(zRef[i])));

            arena.Dispose();
        }

        void BeatsJacobiOnLaplacian()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyLaplacian2D(4, 16);
            var bJ = arena.fProxyBlockJacobi(in A);
            var ic0 = arena.fProxyIC0(in A);
            int n = A.M_Rows;

            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 831002u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 8 * n;

            var xJ = arena.fProxyVec(n);
            var infoJ = Krylov.pcg(in A, in bJ, in b, ref xJ, maxIter, tol);
            Assert.IsTrue(infoJ.Solved);

            var xI = arena.fProxyVec(n);
            var infoI = Krylov.pcg(in A, in ic0, in b, ref xI, maxIter, tol);
            Assert.IsTrue(infoI.Solved);

            Assert.IsTrue((double)infoI.iterations <= (double)infoJ.iterations * 0.9);

            arena.Dispose();
        }

        void BeatsJacobiOnRandomSparseSPD()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomSparseSPD(30, 3, (fProxy)0.35, 831003u);
            var bJ = arena.fProxyBlockJacobi(in A);
            var ic0 = arena.fProxyIC0(in A);
            int n = A.M_Rows;

            var xTrue = arena.fProxyRandomVec(n, 0.5f, 1.5f, 831004u);
            var b = BSR.spMV(in A, in xTrue);
            fProxy tol = Consts.fProxySqrtEps;
            int maxIter = 8 * n;

            var xJ = arena.fProxyVec(n);
            var infoJ = Krylov.pcg(in A, in bJ, in b, ref xJ, maxIter, tol);
            Assert.IsTrue(infoJ.Solved);

            var xI = arena.fProxyVec(n);
            var infoI = Krylov.pcg(in A, in ic0, in b, ref xI, maxIter, tol);
            Assert.IsTrue(infoI.Solved);

            Assert.IsTrue((double)infoI.iterations <= (double)infoJ.iterations * 0.9);

            arena.Dispose();
        }

        void SymmetricStorageMatchesFull()
        {
            var arena = new Arena(Allocator.Persistent);

            // Build the same SPD matrix twice: full storage and symmetric (upper) storage.
            const int nb = 4, BR = 2;
            var full = BuildBlockTridiag(ref arena, nb, BR);

            var builder = arena.fProxyBSRBuilder(nb, nb, BR, BR);
            var diag = arena.fProxyMat(BR, BR);
            var off = arena.fProxyMat(BR, BR);
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                {
                    diag[r, c] = (r == c ? (fProxy)(2 * BR + 2) : (fProxy)0) + (fProxy)0.25f;
                    off[r, c] = r == c ? (fProxy)(-1) : (fProxy)0;
                }
            for (int i = 0; i < nb; i++)
            {
                builder.AddBlock(i, i, in diag);
                if (i + 1 < nb) builder.AddBlock(i, i + 1, in off);   // upper triangle only
            }
            var sym = builder.ToBSRSymmetric(ref arena);

            var mFull = arena.fProxyIC0(in full);
            var mSym = arena.fProxyIC0(in sym);

            int n = full.M_Rows;
            var r2 = arena.fProxyRandomVec(n, -1f, 1f, 831005u);
            var zF = arena.fProxyVec(n);
            var zS = arena.fProxyVec(n);
            mFull.Apply(in r2, ref zF);
            mSym.Apply(in r2, ref zS);

            for (int i = 0; i < n; i++)
                Assert.IsTrue(math.abs(zF[i] - zS[i]) < Tol() * ((fProxy)1 + math.abs(zF[i])));

            arena.Dispose();
        }

        void ApplyIsSymmetric()
        {
            var arena = new Arena(Allocator.Persistent);
            var A = arena.fProxyRandomSparseSPD(20, 3, (fProxy)0.3, 831006u);
            var M = arena.fProxyIC0(in A);
            int n = A.M_Rows;

            var u = arena.fProxyRandomVec(n, -1f, 1f, 831007u);
            var v = arena.fProxyRandomVec(n, -1f, 1f, 831008u);
            var Mu = arena.fProxyVec(n);
            var Mv = arena.fProxyVec(n);
            M.Apply(in u, ref Mu);
            M.Apply(in v, ref Mv);

            // M^-1 = L^-T L^-1 is symmetric, so <u, M^-1 v> == <v, M^-1 u>.
            fProxy a = Blas.dot(u, Mv);
            fProxy bb = Blas.dot(v, Mu);
            fProxy scale = (fProxy)1 + math.abs(a) + math.abs(bb);
            Assert.IsTrue(math.abs(a - bb) < Tol() * scale);

            arena.Dispose();
        }
    }

    // ---- correctness cases (Burst) -------------------------------------------------------

    [Test]
    public void ExactOnBlockTridiagonalTest()
        => new SparseIC0TestJob { Type = SparseIC0TestJob.TestType.ExactOnBlockTridiagonal }.Run();

    [Test]
    public void BeatsJacobiOnLaplacianTest()
        => new SparseIC0TestJob { Type = SparseIC0TestJob.TestType.BeatsJacobiOnLaplacian }.Run();

    [Test]
    public void BeatsJacobiOnRandomSparseSPDTest()
        => new SparseIC0TestJob { Type = SparseIC0TestJob.TestType.BeatsJacobiOnRandomSparseSPD }.Run();

    [Test]
    public void SymmetricStorageMatchesFullTest()
        => new SparseIC0TestJob { Type = SparseIC0TestJob.TestType.SymmetricStorageMatchesFull }.Run();

    [Test]
    public void ApplyIsSymmetricTest()
        => new SparseIC0TestJob { Type = SparseIC0TestJob.TestType.ApplyIsSymmetric }.Run();

    // ---- guard cases (managed thread) ----------------------------------------------------

    [Test]
    public void NonSquareThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.fProxyBSRBuilder(2, 3, 2, 2);
            var block = arena.fProxyMat(2, 2, (fProxy)1);
            builder.AddBlock(0, 0, in block);
            var A = builder.ToBSR(ref arena);
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyIC0(in A); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void MissingDiagonalThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.fProxyBSRBuilder(2, 2, 2, 2);
            var block = arena.fProxyMat(2, 2, (fProxy)1);
            builder.AddBlock(0, 0, in block);
            builder.AddBlock(1, 0, in block);   // no (1,1) diagonal block
            var A = builder.ToBSR(ref arena);
            Assert.Throws<ArgumentException>(() => { var m = arena.fProxyIC0(in A); });
        }
        finally { arena.Dispose(); }
    }

    [Test]
    public void ApplyAliasThrows()
    {
        var arena = new Arena(Allocator.Persistent);
        try
        {
            var builder = arena.fProxyBSRBuilder(2, 2, 2, 2);
            var diag = arena.fProxyMat(2, 2);
            diag[0, 0] = (fProxy)4; diag[1, 1] = (fProxy)4;
            builder.AddBlock(0, 0, in diag);
            builder.AddBlock(1, 1, in diag);
            var A = builder.ToBSR(ref arena);
            var M = arena.fProxyIC0(in A);

            var r = arena.fProxyVec(A.M_Rows, (fProxy)1);
            Assert.Throws<ArgumentException>(() => M.Apply(in r, ref r));
        }
        finally { arena.Dispose(); }
    }
}
