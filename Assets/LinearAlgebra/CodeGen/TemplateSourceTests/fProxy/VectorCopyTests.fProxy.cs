using System;

using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;

// Value-preservation check for vector Copy()/TempCopy(): both must return an independent copy
// whose contents match the source.
public class fProxyVectorCopyTests
{
    [Test]
    public void Copy_IsPersistent_TempCopy_IsTemp()
    {
        var v = new fProxyN(4, Allocator.Temp);
        v[0] = (fProxy)1; v[1] = (fProxy)2; v[2] = (fProxy)3; v[3] = (fProxy)4;

        var c = v.Copy();
        var t = v.TempCopy();

        for (int i = 0; i < 4; i++)
            Assert.AreEqual((double)(i + 1), (double)c[i], 1e-5);
    }
}
