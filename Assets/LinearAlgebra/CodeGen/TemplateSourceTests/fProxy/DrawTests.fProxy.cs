using BULA;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

//+deleteThis
using fProxy2 = Unity.Mathematics.float2;
using fProxy3 = Unity.Mathematics.float3;
//-deleteThis

// Smoke coverage for the Draw wireframe helpers. Debug.DrawLine is void and its output is not
// observable from a test, so these assert DoesNotThrow -- the same treatment Print.Spy and
// Print.Log(in fProxyBSR) get in DebugPrintTests, and for the same reason.
//
// What that still catches is worth having: every method builds an orthonormal frame from a
// caller-supplied direction and loops segment counts, so a bad basis, a divide by a zero-length axis
// or a non-positive segment count would throw or produce NaN here. The DEGENERATE cases below are the
// point of the file -- zero-length directions, a capsule whose endpoints coincide, and segment counts
// under the minimum are all reachable from a fit that did not converge, which is exactly when someone
// reaches for a debug draw.
//
// Managed thread only: Draw is not Burst-callable, like the rest of the Print/Export surface.
public class fProxyDrawTests
{
    static NativeArray<fProxy3> Cloud(int n)
    {
        var pts = new NativeArray<fProxy3>(n, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(17u);
        for (int i = 0; i < n; i++)
            pts[i] = new fProxy3((fProxy)rng.NextDouble(-2.0, 2.0),
                                 (fProxy)rng.NextDouble(-2.0, 2.0),
                                 (fProxy)rng.NextDouble(-2.0, 2.0));
        return pts;
    }

    [Test]
    public void EveryShapeDraws()
    {
        var pts = Cloud(20);
        var axis = new fProxy3((fProxy)0.3, (fProxy)1, (fProxy)(-0.2));
        var origin = new fProxy3((fProxy)1, (fProxy)2, (fProxy)3);

        Assert.DoesNotThrow(() =>
        {
            Draw.points(pts, Color.grey);
            Draw.line(origin, axis, (fProxy)4, Color.cyan);
            Draw.plane(origin, axis, (fProxy)3, Color.blue);
            Draw.circle(origin, axis, (fProxy)2, Color.white);
            Draw.sphere(origin, (fProxy)1.5, Color.green);
            Draw.cylinder(origin, axis, (fProxy)1, (fProxy)3, Color.yellow);
            Draw.cone(origin, axis, (fProxy)0.4, (fProxy)3, Color.magenta);
            Draw.torus(origin, axis, (fProxy)3, (fProxy)0.8, Color.blue);
            Draw.capsule(origin, origin + (fProxy)4 * axis, (fProxy)1, Color.red);
            Draw.ellipse(new fProxy2((fProxy)1, (fProxy)2), new fProxy2((fProxy)3, (fProxy)1),
                         (fProxy)0.5, Color.white);
        });

        pts.Dispose();
    }

    // An unset colour must become visible rather than silently transparent, and every shape must
    // accept `default` without special-casing at the call site.
    [Test]
    public void DefaultColourDraws()
    {
        var origin = new fProxy3((fProxy)0, (fProxy)0, (fProxy)0);
        var axis = new fProxy3((fProxy)0, (fProxy)1, (fProxy)0);

        Assert.DoesNotThrow(() =>
        {
            Draw.sphere(origin, (fProxy)1);
            Draw.circle(origin, axis, (fProxy)1);
            Draw.cylinder(origin, axis, (fProxy)1, (fProxy)2);
        });

        Assert.AreEqual(Color.white, Draw.Resolve(default), "an unset colour must resolve to white");
        Assert.AreEqual(Color.red, Draw.Resolve(Color.red), "an explicit colour must pass through");
    }

    // Degenerate inputs a non-converged fit can hand these: zero-length axes (the frame cannot be
    // built from them), a capsule collapsed to a point, and sub-minimum segment counts.
    [Test]
    public void DegenerateInputsDoNotThrow()
    {
        var zero = new fProxy3((fProxy)0, (fProxy)0, (fProxy)0);
        var p = new fProxy3((fProxy)1, (fProxy)1, (fProxy)1);

        Assert.DoesNotThrow(() =>
        {
            Draw.line(p, zero, (fProxy)2);                       // no direction
            Draw.plane(p, zero, (fProxy)2);
            Draw.circle(p, zero, (fProxy)1);
            Draw.cylinder(p, zero, (fProxy)1, (fProxy)2);
            Draw.capsule(p, p, (fProxy)1);                       // endpoints coincide -> sphere
            Draw.torus(p, zero, (fProxy)2, (fProxy)0.5);
            Draw.circle(p, new fProxy3((fProxy)0, (fProxy)1, (fProxy)0), (fProxy)1, Color.white, 1);
            Draw.sphere(p, (fProxy)1, Color.white, 0);           // segment count below the minimum
            Draw.ellipse(new fProxy2((fProxy)0, (fProxy)0), new fProxy2((fProxy)1, (fProxy)1),
                         (fProxy)0, Color.white, 2);
        });
    }

    // The RANSAC companion: colours points by whether the model accepts them.
    [Test]
    public void ConsensusDrawsAgainstAModel()
    {
        var pts = Cloud(30);
        var model = new Fit.fProxyPlaneModel
        {
            Point = new fProxy3((fProxy)0, (fProxy)0, (fProxy)0),
            Normal = new fProxy3((fProxy)0, (fProxy)0, (fProxy)1),
        };

        Assert.DoesNotThrow(() => Draw.consensus(pts, in model, (fProxy)0.5));
        Assert.DoesNotThrow(() => Draw.consensus(pts, in model, (fProxy)0.5, Color.cyan, Color.yellow));

        pts.Dispose();
    }
}
