using LinearAlgebra;
using NUnit.Framework;
using Unity.Collections;

// Scale-behavior tests for the Frisch-Newton LAD solver. The L1 fit is exactly equivariant in the
// response, so the solver must return the same answer on data that differs only by a positive
// scale factor. Run on the managed test thread (like LadFrischNewtonQuantileTests) so a divergence
// surfaces with a clear message.
public class fProxyLadFrischNewtonScaleTests
{
    // 5 points on b = t with one gross outlier at t=2 (b=10). The LAD fit is (0, 1) with L1 residual
    // 8; the ORDINARY LEAST-SQUARES fit of the same data is (1.6, 1) with L1 residual 12.8. The two
    // are far apart, and the solver starts from the least-squares fit -- so a solve that stops at its
    // starting point instead of iterating is unmistakable here.
    static void BuildOutlierLine(fProxyMxN A, fProxyN b, double scale)
    {
        for (int i = 0; i < 5; i++) { A[i, 0] = (fProxy)1; A[i, 1] = (fProxy)i; b[i] = (fProxy)(i * scale); }
        b[2] = (fProxy)(10.0 * scale);
    }

    // argmin ||A x - c*b||_1 == c * argmin ||A x - b||_1 for every c > 0, and the optimal objective
    // scales by c. Scales span both sides of the z/w initialization floor.
    [Test]
    public void ResponseScaleEquivariance()
    {
        double small = /*+choose[1e-8|1e-16]*/1e-8/*-choose*/;
        double large = /*+choose[1e6|1e12]*/1e6/*-choose*/;
        double relTol = /*+choose[2e-2|1e-6]*/2e-2/*-choose*/;

        var A = new fProxyMxN(5, 2, Allocator.Temp);
        var bUnit = new fProxyN(5, Allocator.Temp);
        var xUnit = new fProxyN(2, Allocator.Temp);
        BuildOutlierLine(A, bUnit, 1.0);
        LP.ladFN(in A, in bUnit, ref xUnit, out double objUnit);

        // Anchor: the unit-scale fit is the known LAD line, not the least-squares line.
        Assert.That(objUnit, Is.EqualTo(8.0).Within(1e-1), "unit-scale L1 residual");

        // Every scale is measured before asserting, so one bad scale does not hide the others.
        double[] scales = { small, 1e-3, 1e3, large };
        var bad = new System.Text.StringBuilder();
        for (int k = 0; k < scales.Length; k++)
        {
            double c = scales[k];
            var bc = new fProxyN(5, Allocator.Temp);
            var xc = new fProxyN(2, Allocator.Temp);
            BuildOutlierLine(A, bc, c);
            LP.ladFN(in A, in bc, ref xc, out double objC);

            double o = objC / c, x0 = (double)xc[0] / c, x1 = (double)xc[1] / c;
            if (System.Math.Abs(o - objUnit) > relTol * objUnit)
                bad.AppendLine($"  scale {c:E0}: objective/c = {o}, expected {objUnit}");
            if (System.Math.Abs(x0 - (double)xUnit[0]) > relTol)
                bad.AppendLine($"  scale {c:E0}: intercept/c = {x0}, expected {(double)xUnit[0]}");
            if (System.Math.Abs(x1 - (double)xUnit[1]) > relTol)
                bad.AppendLine($"  scale {c:E0}: slope/c = {x1}, expected {(double)xUnit[1]}");

            bc.Dispose(); xc.Dispose();
        }
        Assert.That(bad.ToString(), Is.Empty, "LAD fit is not equivariant under b -> c*b:\n" + bad);

        A.Dispose(); bUnit.Dispose(); xUnit.Dispose();
    }
}
