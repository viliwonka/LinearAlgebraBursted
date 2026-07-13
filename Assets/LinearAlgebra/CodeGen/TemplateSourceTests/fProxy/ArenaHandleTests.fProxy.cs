using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Regression suite: allocating through an `in Arena` parameter forces a compiler-inserted
// defensive copy (whenever a non-readonly allocator method like fProxyVec/fProxyMat is called
// through it); the returned struct's Arena handle must still resolve to a live core after the
// helper's frame has returned. These tests reproduce that exact mechanism (allocate via an
// `in Arena` helper, use the result after the helper's frame has returned) inside a
// [BurstCompile] IJob.
public class fProxyArenaHandleTests
{
    [BurstCompile(CompileSynchronously = true)]
    public struct InArenaDanglingTestJob : IJob
    {
        public enum TestType
        {
            Vector,
            Matrix,
            TwoHopVector,
        }

        public TestType Type;

        // Allocate a vector through an `in Arena` parameter. `arena.fProxyVec(...)` is a NON-readonly
        // instance call on the `in` parameter, so the compiler must defensively copy `arena` before
        // the call. Under the old Arena*-capture design the returned vector's arena pointer would
        // capture the address of that now-dead defensive copy once this method returns.
        static fProxyN AllocateVecViaInArena(in Arena arena, int n)
            => arena.fProxyVec(n, (fProxy)0);

        static fProxyMxN AllocateMatViaInArena(in Arena arena, int rows, int cols)
            => arena.fProxyMat(rows, cols, (fProxy)0);

        // Two defensive-copy hops: an `in Arena` helper that forwards to ANOTHER `in Arena` helper.
        // Each `in`-param -> non-readonly-call boundary is its own defensive-copy site, so the old
        // design would have captured (and returned a struct pointing at) the innermost dead temporary.
        static fProxyN AllocateVecTwoHops(in Arena arena, int n)
            => AllocateVecViaInArena(in arena, n);

        public void Execute()
        {
            switch (Type)
            {
                case TestType.Vector: Vector(); break;
                case TestType.Matrix: Matrix(); break;
                case TestType.TwoHopVector: TwoHopVector(); break;
            }
        }

        // ---- vector: allocate via `in Arena` helper, then use only after the helper returned -------
        void Vector()
        {
            var arena = new Arena(Allocator.Persistent);

            const int N = 8;
            // The helper's frame -- and its defensive copy of `arena` -- are gone by the time this
            // returns. `v` now holds the arena handle it captured inside the helper.
            fProxyN v = AllocateVecViaInArena(in arena, N);

            // Indexing writes through v's captured handle-backed buffer.
            for (int i = 0; i < N; i++) v[i] = (fProxy)(i + 1);

            // Copy() allocates a fresh tracked vector THROUGH v's captured Arena handle -- exactly the
            // deref that crashed / read garbage under the old &arena-capture design.
            fProxyN copy = v.Copy();
            for (int i = 0; i < N; i++)
                Assert.IsTrue(v[i] == copy[i]);

            // Both buffers must resolve to the same live core's persistent pool.
            Assert.IsTrue(arena.isPersistent(in v));
            Assert.IsTrue(arena.isPersistent(in copy));
            Assert.AreEqual(2, arena.AllocationsCount); // v + copy, both tracked in the one live core

            arena.Dispose();
        }

        // ---- matrix variant: same mechanism on fProxyMxN / fProxyMat / Copy() ----------------------
        void Matrix()
        {
            var arena = new Arena(Allocator.Persistent);

            const int R = 3, C = 4;
            fProxyMxN m = AllocateMatViaInArena(in arena, R, C);

            for (int r = 0; r < R; r++)
                for (int c = 0; c < C; c++)
                    m[r, c] = (fProxy)(r * C + c + 1);

            fProxyMxN copy = m.Copy();
            Assert.IsTrue(copy.M_Rows == R);
            Assert.IsTrue(copy.N_Cols == C);
            for (int r = 0; r < R; r++)
                for (int c = 0; c < C; c++)
                    Assert.IsTrue(m[r, c] == copy[r, c]);

            Assert.IsTrue(arena.isPersistent(in m));
            Assert.IsTrue(arena.isPersistent(in copy));
            Assert.AreEqual(2, arena.AllocationsCount);

            arena.Dispose();
        }

        // ---- two-hop variant: allocation crosses TWO nested `in Arena` defensive-copy boundaries ---
        void TwoHopVector()
        {
            var arena = new Arena(Allocator.Persistent);

            const int N = 6;
            fProxyN v = AllocateVecTwoHops(in arena, N);

            for (int i = 0; i < N; i++) v[i] = (fProxy)(i + 1);

            fProxyN copy = v.Copy();
            for (int i = 0; i < N; i++)
                Assert.IsTrue(v[i] == copy[i]);

            Assert.IsTrue(arena.isPersistent(in v));
            Assert.IsTrue(arena.isPersistent(in copy));

            arena.Dispose();
        }
    }

    // `in Arena` parameter -> defensive copy -> returned vector's handle must still be live.
    [Test]
    public void InArenaParameter_Vector_DefensiveCopyDoesNotDangle()
        => new InArenaDanglingTestJob { Type = InArenaDanglingTestJob.TestType.Vector }.Run();

    // Matrix path (fProxyMat / fProxyMxN.Copy()): same in-Arena defensive-copy guard.
    [Test]
    public void InArenaParameter_Matrix_DefensiveCopyDoesNotDangle()
        => new InArenaDanglingTestJob { Type = InArenaDanglingTestJob.TestType.Matrix }.Run();

    // Two nested `in Arena` hops (two defensive-copy sites) -- maximally robust guard.
    [Test]
    public void InArenaParameter_TwoHops_DefensiveCopyDoesNotDangle()
        => new InArenaDanglingTestJob { Type = InArenaDanglingTestJob.TestType.TwoHopVector }.Run();
}
