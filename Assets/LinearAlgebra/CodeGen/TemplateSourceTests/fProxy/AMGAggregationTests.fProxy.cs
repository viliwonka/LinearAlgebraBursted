using System;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// AMG.aggregate: deterministic greedy nodal aggregation over BSR. Cases run inside a
// [BurstCompile] IJob. Invariants (not brittle exact-structure): valid partition, coarsening,
// singletons for isolated nodes, theta strength-filtering, and determinism.
public class fProxyAMGAggregationTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct AggregateTestJob : IJob
    {
        public enum TestType
        {
            ChainPartitionsAndCoarsens,
            DiagonalOnlyAllSingletons,
            HighThetaWeakLinksAllSingletons,
            ZeroDiagonalNodeIsolated,
            Deterministic,
        }

        public TestType Type;

        // 1x1-block chain: diagonal 2, symmetric off-diagonal -offMag to each neighbor (full storage).
        static fProxyBSR Chain(int nb, fProxy offMag)
        {
            var b = new fProxyBSRBuilder(nb, nb, 1, 1, Allocator.Temp, 3 * nb);
            for (int i = 0; i < nb; i++)
            {
                b.AddValue(i, i, (fProxy)2);
                if (i > 0) b.AddValue(i, i - 1, -offMag);
                if (i < nb - 1) b.AddValue(i, i + 1, -offMag);
            }
            return b.ToBSR(Allocator.Temp);
        }

        // Every block-row assigned to some aggregate in [0, numAgg); numAgg >= 1.
        static void AssertValidPartition(in Indices aggId, int nb, int numAgg)
        {
            Assert.IsTrue(numAgg >= 1 && numAgg <= nb);
            for (int i = 0; i < nb; i++)
                Assert.IsTrue(aggId[i] >= 0 && aggId[i] < numAgg);
            // Every aggregate id in [0,numAgg) is actually used (no gaps).
            for (int a = 0; a < numAgg; a++)
            {
                bool used = false;
                for (int i = 0; i < nb && !used; i++) used = aggId[i] == a;
                Assert.IsTrue(used);
            }
        }

        public void Execute()
        {
            switch (Type)
            {
                case TestType.ChainPartitionsAndCoarsens:        ChainPartitionsAndCoarsens(); break;
                case TestType.DiagonalOnlyAllSingletons:         DiagonalOnlyAllSingletons(); break;
                case TestType.HighThetaWeakLinksAllSingletons:   HighThetaWeakLinksAllSingletons(); break;
                case TestType.ZeroDiagonalNodeIsolated:          ZeroDiagonalNodeIsolated(); break;
                case TestType.Deterministic:                     Deterministic(); break;
            }
        }

        void ChainPartitionsAndCoarsens()
        {
            int nb = 12;
            var A = Chain(nb, (fProxy)1);
            var aggId = new Indices(nb, Allocator.Temp);

            AMG.aggregate(in A, (fProxy)0, ref aggId, out int numAgg);

            AssertValidPartition(in aggId, nb, numAgg);
            Assert.IsTrue(numAgg < nb);                 // genuine coarsening on a connected chain

            // At least one aggregate has >= 2 members (strong neighbors co-aggregate).
            int maxSize = 0;
            for (int a = 0; a < numAgg; a++)
            {
                int c = 0;
                for (int i = 0; i < nb; i++) if (aggId[i] == a) c++;
                if (c > maxSize) maxSize = c;
            }
            Assert.IsTrue(maxSize >= 2);
        }

        void DiagonalOnlyAllSingletons()
        {
            int nb = 8;
            var b = new fProxyBSRBuilder(nb, nb, 1, 1, Allocator.Temp, nb);
            for (int i = 0; i < nb; i++) b.AddValue(i, i, (fProxy)3);
            var A = b.ToBSR(Allocator.Temp);
            var aggId = new Indices(nb, Allocator.Temp);

            AMG.aggregate(in A, (fProxy)0, ref aggId, out int numAgg);

            AssertValidPartition(in aggId, nb, numAgg);
            Assert.IsTrue(numAgg == nb);                // no connections -> every node its own aggregate
        }

        void HighThetaWeakLinksAllSingletons()
        {
            int nb = 10;
            var A = Chain(nb, (fProxy)0.01);   // off-diagonals tiny vs diagonal 2
            var aggId = new Indices(nb, Allocator.Temp);

            // threshold = theta*sqrt(2*2) = 1.0 >> 0.01 -> all links weak -> all singletons.
            AMG.aggregate(in A, (fProxy)0.5, ref aggId, out int numAgg);

            AssertValidPartition(in aggId, nb, numAgg);
            Assert.IsTrue(numAgg == nb);

            // Same matrix with theta = 0 keeps the links -> coarsens.
            var aggId0 = new Indices(nb, Allocator.Temp);
            AMG.aggregate(in A, (fProxy)0, ref aggId0, out int numAgg0);
            Assert.IsTrue(numAgg0 < nb);
        }

        // A zero-diagonal block (constraint/interface row) has undefined strength; with theta>0 its
        // edges must be treated WEAK so it falls through to a pass-3 singleton, NOT pulled into a
        // real-DOF aggregate. (Regression for the strength-normalizer bug where theta*sqrt(0*d)==0
        // made every incident edge spuriously strong.)
        void ZeroDiagonalNodeIsolated()
        {
            int nb = 5;
            // Chain with node 2 given a ZERO diagonal; off-diagonals -1 everywhere.
            var b = new fProxyBSRBuilder(nb, nb, 1, 1, Allocator.Temp, 3 * nb);
            for (int i = 0; i < nb; i++)
            {
                b.AddValue(i, i, i == 2 ? (fProxy)0 : (fProxy)2);
                if (i > 0) b.AddValue(i, i - 1, (fProxy)(-1));
                if (i < nb - 1) b.AddValue(i, i + 1, (fProxy)(-1));
            }
            var A = b.ToBSR(Allocator.Temp);
            var aggId = new Indices(nb, Allocator.Temp);

            // theta=0.5: real-DOF edges (|off|=1 vs threshold 0.5*sqrt(2)*sqrt(2)≈1) are strong;
            // edges incident to the zero-diagonal node 2 are weak.
            AMG.aggregate(in A, (fProxy)0.5, ref aggId, out int numAgg);

            AssertValidPartition(in aggId, nb, numAgg);

            // node 2's aggregate must contain only node 2 (singleton).
            int a2 = aggId[2];
            int size = 0;
            for (int i = 0; i < nb; i++) if (aggId[i] == a2) size++;
            Assert.IsTrue(size == 1);
        }

        void Deterministic()
        {
            int nb = 16;
            var A = Chain(nb, (fProxy)1);
            var a1 = new Indices(nb, Allocator.Temp);
            var a2 = new Indices(nb, Allocator.Temp);

            AMG.aggregate(in A, (fProxy)0, ref a1, out int n1);
            AMG.aggregate(in A, (fProxy)0, ref a2, out int n2);

            Assert.IsTrue(n1 == n2);
            for (int i = 0; i < nb; i++) Assert.IsTrue(a1[i] == a2[i]);
        }
    }

    [Test]
    public void ChainPartitionsAndCoarsensTest()
        => new AggregateTestJob { Type = AggregateTestJob.TestType.ChainPartitionsAndCoarsens }.Run();

    [Test]
    public void DiagonalOnlyAllSingletonsTest()
        => new AggregateTestJob { Type = AggregateTestJob.TestType.DiagonalOnlyAllSingletons }.Run();

    [Test]
    public void HighThetaWeakLinksAllSingletonsTest()
        => new AggregateTestJob { Type = AggregateTestJob.TestType.HighThetaWeakLinksAllSingletons }.Run();

    [Test]
    public void ZeroDiagonalNodeIsolatedTest()
        => new AggregateTestJob { Type = AggregateTestJob.TestType.ZeroDiagonalNodeIsolated }.Run();

    [Test]
    public void DeterministicTest()
        => new AggregateTestJob { Type = AggregateTestJob.TestType.Deterministic }.Run();
}
