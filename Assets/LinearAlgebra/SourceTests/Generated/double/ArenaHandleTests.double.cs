using LinearAlgebra;
using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

// Regression suite for failure mode 2 (FM2) of the Arena memory-model fix
// (docs/dev/rfc-memory-model.md §1 / §2.2 / §4 Option A / §6.0 / §6.1).
//
// THE OLD BUG (FM2): Arena used to be a plain struct holding all its mutable tracking state inline,
// and every math struct captured arena identity by RAW ADDRESS (`Arena* _arenaPtr`, set via
// `fixed (Arena* p = &arena) _arenaPtr = p;` in the `in Arena` constructors). Arena's allocator
// methods (e.g. doubleVec / doubleMat) are NOT `readonly`, so calling one through an `in Arena`
// PARAMETER forces the C# compiler to make a defensive copy of the arena first -- and the struct
// being constructed captured the address of that dead stack temporary. Once the enclosing helper's
// frame returned, that captured pointer dangled: indexing/Copy()/Dispose() on the returned struct
// dereferenced freed stack memory, surfacing under Burst as a native crash / "allocator handle is
// not valid".
//
// THE FIX: Arena was split into ArenaCore (heap-Malloc'd once, holds all mutable state, never copied)
// and Arena (a thin handle wrapping a single `ArenaCore*`). Every math struct now holds an Arena
// VALUE field. Copying an Arena handle -- including compiler-inserted defensive copies of `in Arena`
// params -- copies only the ArenaCore* value, so every copy still resolves to the same live core.
// The dangling-pointer failure mode is now structurally impossible.
//
// These tests reproduce the EXACT mechanism: allocation happens through an `in Arena` PARAMETER
// inside a helper that calls a mutating Arena allocator method (forcing the historical defensive
// copy), and the returned struct is only USED (indexed, Copy()'d, isPersistent-checked, disposed)
// AFTER that helper's stack frame has returned. They run inside a [BurstCompile] IJob because that is
// where the original bug actually manifested as a native crash -- a pure managed-thread test might
// not reproduce it even with the old bug present.
public class doubleArenaHandleTests
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

        // Allocate a vector through an `in Arena` parameter. `arena.doubleVec(...)` is a NON-readonly
        // instance call on the `in` parameter, so the compiler must defensively copy `arena` before
        // the call. Under the old Arena*-capture design the returned vector's arena pointer would
        // capture the address of that now-dead defensive copy once this method returns (FM2).
        static doubleN AllocateVecViaInArena(in Arena arena, int n)
            => arena.doubleVec(n, (double)0);

        static doubleMxN AllocateMatViaInArena(in Arena arena, int rows, int cols)
            => arena.doubleMat(rows, cols, (double)0);

        // Two defensive-copy hops: an `in Arena` helper that forwards to ANOTHER `in Arena` helper.
        // Each `in`-param -> non-readonly-call boundary is its own defensive-copy site, so the old
        // design would have captured (and returned a struct pointing at) the innermost dead temporary.
        static doubleN AllocateVecTwoHops(in Arena arena, int n)
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
            doubleN v = AllocateVecViaInArena(in arena, N);

            // Indexing writes through v's captured handle-backed buffer.
            for (int i = 0; i < N; i++) v[i] = (double)(i + 1);

            // Copy() allocates a fresh tracked vector THROUGH v's captured Arena handle -- exactly the
            // deref that crashed / read garbage under the old &arena-capture design.
            doubleN copy = v.Copy();
            for (int i = 0; i < N; i++)
                Assert.IsTrue(v[i] == copy[i]);

            // Both buffers must resolve to the same live core's persistent pool.
            Assert.IsTrue(arena.isPersistent(in v));
            Assert.IsTrue(arena.isPersistent(in copy));
            Assert.AreEqual(2, arena.AllocationsCount); // v + copy, both tracked in the one live core

            arena.Dispose();
        }

        // ---- matrix variant: same mechanism on doubleMxN / doubleMat / Copy() ----------------------
        void Matrix()
        {
            var arena = new Arena(Allocator.Persistent);

            const int R = 3, C = 4;
            doubleMxN m = AllocateMatViaInArena(in arena, R, C);

            for (int r = 0; r < R; r++)
                for (int c = 0; c < C; c++)
                    m[r, c] = (double)(r * C + c + 1);

            doubleMxN copy = m.Copy();
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
            doubleN v = AllocateVecTwoHops(in arena, N);

            for (int i = 0; i < N; i++) v[i] = (double)(i + 1);

            doubleN copy = v.Copy();
            for (int i = 0; i < N; i++)
                Assert.IsTrue(v[i] == copy[i]);

            Assert.IsTrue(arena.isPersistent(in v));
            Assert.IsTrue(arena.isPersistent(in copy));

            arena.Dispose();
        }
    }

    // FM2: `in Arena` parameter -> defensive copy -> returned vector's handle must still be live.
    [Test]
    public void InArenaParameter_Vector_DefensiveCopyDoesNotDangle()
        => new InArenaDanglingTestJob { Type = InArenaDanglingTestJob.TestType.Vector }.Run();

    // FM2, matrix path (doubleMat / doubleMxN.Copy()).
    [Test]
    public void InArenaParameter_Matrix_DefensiveCopyDoesNotDangle()
        => new InArenaDanglingTestJob { Type = InArenaDanglingTestJob.TestType.Matrix }.Run();

    // FM2 through two nested `in Arena` hops (two defensive-copy sites) -- maximally robust.
    [Test]
    public void InArenaParameter_TwoHops_DefensiveCopyDoesNotDangle()
        => new InArenaDanglingTestJob { Type = InArenaDanglingTestJob.TestType.TwoHopVector }.Run();
}
