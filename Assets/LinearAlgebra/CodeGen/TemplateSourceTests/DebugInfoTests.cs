using LinearAlgebra;

using NUnit.Framework;
using Unity.Collections;

// Content-correctness tests for the MANAGED-side ToString() of the small info/result structs
// (SolveInfo / LstsqInfo / DirectSolveInfo / RankRevealingInfo / EigenSolveInfo / LanczosInfo,
// OP/Solvers.Info.cs + OP/Eigen.Info.cs) and the Pivot / Indices permutation types
// (Pivot/Pivot.cs, Indices/Indices.cs). These types are NOT templated (type-agnostic; their
// numbers are reported as plain int/double), so this test file is a plain copy-through (no
// per-type placeholder in the filename or body) -- it lands in SourceTests/Generated unchanged.
//
// ToString() wraps the Burst ToFixedString() and is readable on the managed thread, so every
// assertion here is plain managed C# (no Burst job): build a struct, call ToString(), assert the
// string is RIGHT. The void Print.Log(...) overloads (Debug/Debug.Info.cs) are Burst-void
// log-only, so they only get DoesNotThrow smoke coverage (same pattern as DebugExportTests'
// IntLogDoesNotThrow).
public class DebugInfoTests
{
    // ---------------- SolveInfo (square Krylov solve) ----------------

    [Test]
    public void SolveInfo_Converged_ToStringHasStatusItersRnorm()
    {
        var info = new SolveInfo
        {
            status = IterativeSolveStatus.Converged,
            iterations = 42,
            rnorm = 1.23e-8,
        };

        string s = info.ToString();
        StringAssert.StartsWith("SolveInfo(", s);
        StringAssert.Contains("Converged", s);
        StringAssert.Contains("iters=42", s);
        StringAssert.Contains("rnorm=", s);   // G3-formatted value follows the label
        StringAssert.EndsWith(")", s);
    }

    // Non-Converged status exercises the status-name switch's MaxIterations branch.
    [Test]
    public void SolveInfo_MaxIterations_ToStringNamesTheStatus()
    {
        var info = new SolveInfo
        {
            status = IterativeSolveStatus.MaxIterations,
            iterations = 1000,
            rnorm = 5.0,
        };

        string s = info.ToString();
        StringAssert.Contains("MaxIterations", s);
        StringAssert.Contains("iters=1000", s);
        Assert.IsFalse(s.Contains("Converged"));
    }

    // ---------------- LstsqInfo (least-squares Krylov solve) ----------------

    [Test]
    public void LstsqInfo_Converged_ToStringHasAllThreeNorms()
    {
        var info = new LstsqInfo
        {
            status = IterativeSolveStatus.Converged,
            iterations = 17,
            rnorm = 1.23e-8,
            Arnorm = 4.56e-9,
            xnorm = 2.10,
        };

        string s = info.ToString();
        StringAssert.StartsWith("LstsqInfo(", s);
        StringAssert.Contains("Converged", s);
        StringAssert.Contains("iters=17", s);
        StringAssert.Contains("rnorm=", s);
        StringAssert.Contains("Arnorm=", s);
        StringAssert.Contains("xnorm=", s);
    }

    [Test]
    public void LstsqInfo_Breakdown_ToStringNamesTheStatus()
    {
        var info = new LstsqInfo
        {
            status = IterativeSolveStatus.Breakdown,
            iterations = 3,
        };

        StringAssert.Contains("Breakdown", info.ToString());
    }

    // ---------------- DirectSolveInfo (LU / plain Cholesky / triangular solves) ----------------

    [Test]
    public void DirectSolveInfo_Success_ToStringIsExact()
    {
        var info = new DirectSolveInfo { status = DirectSolveStatus.Success };
        Assert.AreEqual("DirectSolveInfo(Success)", info.ToString());
    }

    // A hard-failure status exercises the DirectSolveStatus name switch's Singular branch.
    [Test]
    public void DirectSolveInfo_Singular_ToStringIsExact()
    {
        var info = new DirectSolveInfo { status = DirectSolveStatus.Singular };
        Assert.AreEqual("DirectSolveInfo(Singular)", info.ToString());
    }

    // ---------------- RankRevealingInfo (QRCP / pivoted Cholesky) ----------------

    [Test]
    public void RankRevealingInfo_RankDeficient_ToStringIsExact()
    {
        var info = new RankRevealingInfo { status = DirectSolveStatus.RankDeficient, rank = 3 };
        Assert.AreEqual("RankRevealingInfo(RankDeficient, rank=3)", info.ToString());
    }

    [Test]
    public void RankRevealingInfo_Success_ToStringCarriesRank()
    {
        var info = new RankRevealingInfo { status = DirectSolveStatus.Success, rank = 5 };
        string s = info.ToString();
        StringAssert.Contains("Success", s);
        StringAssert.Contains("rank=5", s);
    }

    // ---------------- EigenSolveInfo (power / inverse-power iteration) ----------------

    [Test]
    public void EigenSolveInfo_Converged_ToStringHasItersResidual()
    {
        var info = new EigenSolveInfo
        {
            status = IterativeSolveStatus.Converged,
            iterations = 12,
            residual = 1.23e-8,
        };

        string s = info.ToString();
        StringAssert.StartsWith("EigenSolveInfo(", s);
        StringAssert.Contains("Converged", s);
        StringAssert.Contains("iters=12", s);
        StringAssert.Contains("residual=", s);
    }

    // Breakdown return carries NaN residual -- interpolation must render it without throwing.
    [Test]
    public void EigenSolveInfo_Breakdown_ToStringNamesStatusAndRendersNaN()
    {
        var info = new EigenSolveInfo
        {
            status = IterativeSolveStatus.Breakdown,
            iterations = 0,
            residual = double.NaN,
        };

        string s = info.ToString();
        StringAssert.Contains("Breakdown", s);
        StringAssert.Contains("NaN", s);
    }

    // ---------------- LanczosInfo (symmetric Lanczos tridiagonalization) ----------------

    [Test]
    public void LanczosInfo_Converged_ToStringIsExact()
    {
        var info = new LanczosInfo { status = IterativeSolveStatus.Converged, produced = 20 };
        Assert.AreEqual("LanczosInfo(Converged, produced=20)", info.ToString());
    }

    [Test]
    public void LanczosInfo_MaxIterations_ToStringNamesStatus()
    {
        var info = new LanczosInfo { status = IterativeSolveStatus.MaxIterations, produced = 7 };
        string s = info.ToString();
        StringAssert.Contains("MaxIterations", s);
        StringAssert.Contains("produced=7", s);
    }

    // ---------------- info-struct Print.Log smoke (Burst-void log-only) ----------------

    [Test]
    public void InfoLogsDoNotThrow()
    {
        var solve = new SolveInfo { status = IterativeSolveStatus.Converged, iterations = 1 };
        var lstsq = new LstsqInfo { status = IterativeSolveStatus.MaxIterations, iterations = 2 };
        var direct = new DirectSolveInfo { status = DirectSolveStatus.Success };
        var rank = new RankRevealingInfo { status = DirectSolveStatus.RankDeficient, rank = 3 };
        var eigen = new EigenSolveInfo { status = IterativeSolveStatus.Converged, iterations = 4, residual = 1e-9 };
        var lanczos = new LanczosInfo { status = IterativeSolveStatus.Converged, produced = 5 };

        Assert.DoesNotThrow(() => Print.Log(in solve));
        Assert.DoesNotThrow(() => Print.Log(in lstsq));
        Assert.DoesNotThrow(() => Print.Log(in direct));
        Assert.DoesNotThrow(() => Print.Log(in rank));
        Assert.DoesNotThrow(() => Print.Log(in eigen));
        Assert.DoesNotThrow(() => Print.Log(in lanczos));
    }

    // ---------------- Pivot ToString + sign parity ----------------

    // Three effective swaps -> odd parity -> sign -1, and the permutation body reads out exactly.
    [Test]
    public void Pivot_OddSwaps_ToStringExactAndSignNegative()
    {
        var p = new Pivot(5, Allocator.Temp);
        try
        {
            // [0 1 2 3 4] --Swap(0,2)--> [2 1 0 3 4] --Swap(1,2)--> [2 0 1 3 4] --Swap(3,4)--> [2 0 1 4 3]
            p.Swap(0, 2);
            p.Swap(1, 2);
            p.Swap(3, 4);

            Assert.AreEqual(-1, p.Sign);                                     // 3 effective swaps -> odd
            Assert.AreEqual("Pivot[N=5, sign=-1]: (2 0 1 4 3)", p.ToString());
        }
        finally { p.Dispose(); }
    }

    // An even number of effective swaps returns to identity -> sign +1 -> "(0 1 2)".
    [Test]
    public void Pivot_EvenSwaps_ToStringExactAndSignPositive()
    {
        var p = new Pivot(3, Allocator.Temp);
        try
        {
            p.Swap(0, 1);
            p.Swap(0, 1);   // undoes the first -> back to identity, 2 effective swaps -> even

            Assert.AreEqual(1, p.Sign);
            Assert.AreEqual("Pivot[N=3, sign=+1]: (0 1 2)", p.ToString());
        }
        finally { p.Dispose(); }
    }

    // Swap(i,i) is not an effective swap: parity (and the sign field in the string) stays +1.
    [Test]
    public void Pivot_SelfSwapKeepsSignPositive()
    {
        var p = new Pivot(4, Allocator.Temp);
        try
        {
            p.Swap(2, 2);   // no-op for parity
            Assert.AreEqual(1, p.Sign);
            StringAssert.Contains("sign=+1", p.ToString());
        }
        finally { p.Dispose(); }
    }

    [Test]
    public void Pivot_LogDoesNotThrow()
    {
        var p = new Pivot(4, Allocator.Temp);
        try
        {
            p.Swap(1, 3);
            Assert.DoesNotThrow(() => Print.Log(in p));
        }
        finally { p.Dispose(); }
    }

    // ---------------- Indices ToString ----------------

    [Test]
    public void Indices_ToStringIsExact()
    {
        var idx = new Indices(4, Allocator.Temp);
        try
        {
            idx[0] = 7; idx[1] = 2; idx[2] = 9; idx[3] = 0;
            Assert.AreEqual("Indices[N=4]: (7 2 9 0)", idx.ToString());
        }
        finally { idx.Dispose(); }
    }

    [Test]
    public void Indices_LogDoesNotThrow()
    {
        var idx = new Indices(3, Allocator.Temp);
        try
        {
            idx[0] = 1; idx[1] = 4; idx[2] = 9;
            Assert.DoesNotThrow(() => Print.Log(in idx));
        }
        finally { idx.Dispose(); }
    }
}
