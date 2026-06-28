using System;

using LinearAlgebra;

using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
using LinearAlgebra.mathProxies;
//-deleteThis

// Guards for arena.Convert(in fProxyN -> fProxy2): the source vector must have length >= 2,
// otherwise reading [1] would be out of bounds. These are managed [Test]s (main thread) because
// they assert exception behavior; the positive path is a plain element-equality check.
public class fProxyArenaConversionsTests
{
    // Length < 2 must throw ArgumentException (cannot fill .y).
    [Test]
    public void ConvertToVec2TooShortThrows()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.fProxyVec(1);
        v[0] = 7f;

        Assert.Throws<ArgumentException>(() => arena.Convert(in v));

        arena.Dispose();
    }

    // Length == 2 converts; .x/.y mirror v[0]/v[1] exactly.
    [Test]
    public void ConvertToVec2ExactLengthMatches()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.fProxyVec(2);
        v[0] = 3f; v[1] = -5f;

        fProxy2 r = arena.Convert(in v);

        Assert.IsTrue(r.x == v[0]);
        Assert.IsTrue(r.y == v[1]);

        arena.Dispose();
    }

    // Length > 2 also converts, taking only the first two components.
    [Test]
    public void ConvertToVec2LongerVectorTakesFirstTwo()
    {
        var arena = new Arena(Allocator.Persistent);

        var v = arena.fProxyVec(4);
        v[0] = 1f; v[1] = 2f; v[2] = 3f; v[3] = 4f;

        fProxy2 r = arena.Convert(in v);

        Assert.IsTrue(r.x == v[0]);
        Assert.IsTrue(r.y == v[1]);

        arena.Dispose();
    }
}
