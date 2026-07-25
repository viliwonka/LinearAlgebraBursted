using System;

using BULA;

using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

//+deleteThis
// TEMPLATE-ONLY alias, mirrors ConvertOP.fProxy.cs: fProxy2 -> float2/double2 (real
// Unity.Mathematics type), so ConvertOP.Convert(in fProxyN) below binds to the same type its
// return value uses in generated code.
using fProxy2 = Unity.Mathematics.float2;
//-deleteThis

// Guards for ConvertOP.Convert(in fProxyN -> fProxy2): the source vector must have length >= 2,
// otherwise reading [1] would be out of bounds. These are managed [Test]s (main thread) because
// they assert exception behavior; the positive path is a plain element-equality check.
public class fProxyArenaConversionsTests
{
    // Length < 2 must throw ArgumentException (cannot fill .y).
    [Test]
    public void ConvertToVec2TooShortThrows()
    {
        var v = new fProxyN(1, Allocator.Temp);
        v[0] = 7f;

        Assert.Throws<ArgumentException>(() => ConvertOP.Convert(in v));
    }

    // Length == 2 converts; .x/.y mirror v[0]/v[1] exactly.
    [Test]
    public void ConvertToVec2ExactLengthMatches()
    {
        var v = new fProxyN(2, Allocator.Temp);
        v[0] = 3f; v[1] = -5f;

        fProxy2 r = ConvertOP.Convert(in v);

        Assert.IsTrue(r.x == v[0]);
        Assert.IsTrue(r.y == v[1]);
    }

    // Length > 2 also converts, taking only the first two components.
    [Test]
    public void ConvertToVec2LongerVectorTakesFirstTwo()
    {
        var v = new fProxyN(4, Allocator.Temp);
        v[0] = 1f; v[1] = 2f; v[2] = 3f; v[3] = 4f;

        fProxy2 r = ConvertOP.Convert(in v);

        Assert.IsTrue(r.x == v[0]);
        Assert.IsTrue(r.y == v[1]);
    }
}
