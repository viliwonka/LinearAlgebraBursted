using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// AMG.galerkinRAP: the unsmoothed coarse operator A_c = TᵀAT, assembled by segmented scatter-add
// (no spGEMM). Verified matrix-free via the Galerkin identity <v, A_c u> == <Tv, A(Tu)>, plus
// coarse-operator symmetry and determinism. Cases run inside a [BurstCompile] IJob.
public class fProxyAMGGalerkinTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct GalerkinTestJob : IJob
    {
        public enum TestType
        {
            IdentityScalar,
            IdentityBlock,
            CoarseIsSymmetric,
            Deterministic,
        }

        public TestType Type;

        static fProxy Tol() => /*+choose[1e-3f|1e-9]*/1e-3f/*-choose*/;

        static fProxyBSR BlockChain(int nb, int BR)
        {
            var b = new fProxyBSRBuilder(nb, nb, BR, BR, Allocator.Temp, 3 * nb);
            var diag = new fProxyMxN(BR, BR, Allocator.Temp);
            var off = new fProxyMxN(BR, BR, Allocator.Temp);
            for (int r = 0; r < BR; r++)
                for (int c = 0; c < BR; c++)
                { diag[r, c] = r == c ? (fProxy)2 : (fProxy)0; off[r, c] = r == c ? (fProxy)(-1) : (fProxy)0; }
            for (int i = 0; i < nb; i++)
            {
                b.AddBlock(i, i, in diag);
                if (i > 0) b.AddBlock(i, i - 1, in off);
                if (i < nb - 1) b.AddBlock(i, i + 1, in off);
            }
            return b.ToBSR(Allocator.Temp);
        }

        // <v, A_c u> vs <T v, A (T u)>: must agree for A_c = TᵀAT.
        void CheckIdentity(in fProxyBSR A, in fProxyBSR T, in fProxyBSR Ac, int ncoarse, uint seed)
        {
            var u = GenerateOP.fProxyRandomVec(ncoarse, -1f, 1f, seed);
            var v = GenerateOP.fProxyRandomVec(ncoarse, -1f, 1f, seed ^ 0x9E3779B9u);

            var Tu = BSR.spMV(in T, in u);
            var Tv = BSR.spMV(in T, in v);
            var ATu = BSR.spMV(in A, in Tu);
            fProxy lhs = Blas.dot(Tv, ATu);

            var Acu = BSR.spMV(in Ac, in u);
            fProxy rhs = Blas.dot(v, Acu);

            Assert.IsTrue(math.abs(lhs - rhs) <= Tol() * ((fProxy)1 + math.abs(rhs)));
        }

        public void Execute()
        {
            switch (Type)
            {
                case TestType.IdentityScalar:    IdentityScalar(); break;
                case TestType.IdentityBlock:     IdentityBlock(); break;
                case TestType.CoarseIsSymmetric: CoarseIsSymmetric(); break;
                case TestType.Deterministic:     Deterministic(); break;
            }
        }

        void IdentityScalar()
        {
            int nb = 16;
            var A = BlockChain(nb, 1);
            var aggId = new Indices(nb, Allocator.Temp);
            AMG.aggregate(in A, (fProxy)0, ref aggId, out int numAgg);
            var T = AMG.tentativeProlongator(in A, in aggId, numAgg, out _);
            var Ac = AMG.galerkinRAP(in A, in T, in aggId, numAgg);

            Assert.IsTrue(Ac.BlockRows == numAgg && Ac.BR == 1);
            CheckIdentity(in A, in T, in Ac, numAgg * 1, 0x5A1Du);
        }

        void IdentityBlock()
        {
            int nb = 12, BR = 2, m = 2;
            var A = BlockChain(nb, BR);
            int n = nb * BR;
            var B = GenerateOP.fProxyRandomMat(n, m, -1f, 1f, 0x77A1u);
            var aggId = new Indices(nb, Allocator.Temp);
            AMG.aggregate(in A, (fProxy)0, ref aggId, out int numAgg);
            var T = AMG.tentativeProlongator(in A, in aggId, numAgg, in B, out _);
            var Ac = AMG.galerkinRAP(in A, in T, in aggId, numAgg);

            Assert.IsTrue(Ac.BlockRows == numAgg && Ac.BR == m);
            CheckIdentity(in A, in T, in Ac, numAgg * m, 0x1234u);
        }

        // A_c symmetric (A SPD -> TᵀAT SPD): <v, A_c u> == <u, A_c v>.
        void CoarseIsSymmetric()
        {
            int nb = 12, BR = 2, m = 2;
            var A = BlockChain(nb, BR);
            int n = nb * BR;
            var B = GenerateOP.fProxyRandomMat(n, m, -1f, 1f, 0x9001u);
            var aggId = new Indices(nb, Allocator.Temp);
            AMG.aggregate(in A, (fProxy)0, ref aggId, out int numAgg);
            var T = AMG.tentativeProlongator(in A, in aggId, numAgg, in B, out _);
            var Ac = AMG.galerkinRAP(in A, in T, in aggId, numAgg);

            int ncoarse = numAgg * m;
            var u = GenerateOP.fProxyRandomVec(ncoarse, -1f, 1f, 0xAAu);
            var v = GenerateOP.fProxyRandomVec(ncoarse, -1f, 1f, 0xBBu);
            var Acu = BSR.spMV(in Ac, in u);
            var Acv = BSR.spMV(in Ac, in v);
            fProxy a = Blas.dot(v, Acu);
            fProxy b2 = Blas.dot(u, Acv);
            Assert.IsTrue(math.abs(a - b2) <= Tol() * ((fProxy)1 + math.abs(a)));
        }

        void Deterministic()
        {
            int nb = 14, BR = 2, m = 2;
            var A = BlockChain(nb, BR);
            int n = nb * BR;
            var B = GenerateOP.fProxyRandomMat(n, m, -1f, 1f, 0xDEEDu);
            var aggId = new Indices(nb, Allocator.Temp);
            AMG.aggregate(in A, (fProxy)0, ref aggId, out int numAgg);
            var T = AMG.tentativeProlongator(in A, in aggId, numAgg, in B, out _);

            var Ac1 = AMG.galerkinRAP(in A, in T, in aggId, numAgg);
            var Ac2 = AMG.galerkinRAP(in A, in T, in aggId, numAgg);

            Assert.IsTrue(Ac1.Nnzb == Ac2.Nnzb);
            int blockLen = m * m;
            for (int k = 0; k < Ac1.Nnzb; k++) Assert.IsTrue(Ac1.ColInd[k] == Ac2.ColInd[k]);
            for (int k = 0; k < Ac1.Nnzb * blockLen; k++) Assert.IsTrue(Ac1.Values[k] == Ac2.Values[k]);
        }
    }

    [Test]
    public void IdentityScalarTest()
        => new GalerkinTestJob { Type = GalerkinTestJob.TestType.IdentityScalar }.Run();

    [Test]
    public void IdentityBlockTest()
        => new GalerkinTestJob { Type = GalerkinTestJob.TestType.IdentityBlock }.Run();

    [Test]
    public void CoarseIsSymmetricTest()
        => new GalerkinTestJob { Type = GalerkinTestJob.TestType.CoarseIsSymmetric }.Run();

    [Test]
    public void DeterministicTest()
        => new GalerkinTestJob { Type = GalerkinTestJob.TestType.Deterministic }.Run();
}
