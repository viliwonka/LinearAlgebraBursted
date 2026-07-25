using BULA;
using BULA.Sparse;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// Sparse (BSR) Frisch-Newton against the dense engine it shares a core with. The two differ only in
// how the design matrix is stored and how A x, A^T y and the weighted Gram A^T diag(q) A are formed,
// so on the same data they must agree to summation-order noise -- a much tighter contract than
// "sparse LAD is roughly right", and the one that would catch a wrong sparse Gram.
public class fProxyLadSparseDesignTests
{
    static fProxyBSR ToBsr1x1(in fProxyMxN dense, Allocator alloc)
    {
        int m = dense.M_Rows, n = dense.N_Cols;
        int nnz = 0;
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) if (dense[i, j] != (fProxy)0) nnz++;
        var builder = new fProxyBSRBuilder(m, n, 1, 1, alloc, nnz);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (dense[i, j] != (fProxy)0)
                {
                    var blk = new fProxyMxN(1, 1, alloc);
                    blk[0, 0] = dense[i, j];
                    builder.AddBlock(i, j, in blk);
                    blk.Dispose();
                }
        return builder.ToBSR(alloc);
    }

    // Tall design with a genuinely sparse pattern (each row touches 2 of n columns plus an intercept)
    // and gross outliers, so the L1 fit differs sharply from the L2 one and the Gram has real
    // off-diagonal structure rather than being near-diagonal.
    static void BuildSparseDesign(int m, int n, out fProxyMxN A, out fProxyN b, uint seed)
    {
        A = new fProxyMxN(m, n, Allocator.Temp);
        b = new fProxyN(m, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(seed);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++) A[i, j] = (fProxy)0;
            A[i, 0] = (fProxy)1;
            int c1 = 1 + rng.NextInt(0, n - 1);
            int c2 = 1 + rng.NextInt(0, n - 1);
            A[i, c1] = rng.NextFProxy(0.5f, 2f);
            A[i, c2] = rng.NextFProxy(0.5f, 2f);
            b[i] = (fProxy)(2f * (float)A[i, c1] - 1.5f * (float)A[i, c2] + 0.75f + rng.NextFProxy(-0.3f, 0.3f));
        }
        for (int k = 0; k < m / 10; k++) b[k * 7 % m] = (fProxy)((float)b[k * 7 % m] + 25f);
    }

    [Test]
    public void SparseFrischNewtonMatchesDense()
    {
        int[] ms = { 64, 256 };
        int n = 6;
        double relTol = /*+choose[3e-3|1e-9]*/3e-3/*-choose*/;

        var bad = new System.Text.StringBuilder();
        for (int k = 0; k < ms.Length; k++)
        {
            int m = ms[k];
            BuildSparseDesign(m, n, out var A, out var b, 99u + (uint)m);
            var As = ToBsr1x1(in A, Allocator.Temp);

            var xd = new fProxyN(n, Allocator.Temp);
            var xs = new fProxyN(n, Allocator.Temp);
            var infoD = LP.ladFN(in A, in b, ref xd, out double objD);
            var infoS = LP.ladFN(in As, in b, ref xs, out double objS);

            if (System.Math.Abs(objS - objD) > relTol * (1.0 + objD))
                bad.AppendLine($"  m={m}: sparse L1 {objS} vs dense {objD}");
            if (infoS.status != infoD.status)
                bad.AppendLine($"  m={m}: status {infoS.status} vs dense {infoD.status}");
            for (int j = 0; j < n; j++)
                if (System.Math.Abs((double)xs[j] - (double)xd[j]) > relTol * (1.0 + System.Math.Abs((double)xd[j])))
                    bad.AppendLine($"  m={m}: x[{j}] sparse {(double)xs[j]} vs dense {(double)xd[j]}");

            A.Dispose(); b.Dispose(); As.Dispose(); xd.Dispose(); xs.Dispose();
        }
        Assert.That(bad.ToString(), Is.Empty, "sparse Frisch-Newton disagrees with dense:\n" + bad);
    }

    // Same at a non-median tau: the sparse weighted Gram must be tau-blind (tau enters only through
    // the a/s starting values and the box), so any tau-dependent divergence means a storage bug.
    [Test]
    public void SparseQuantileMatchesDense()
    {
        int m = 128, n = 5;
        double[] taus = { 0.25, 0.75 };
        double relTol = /*+choose[3e-3|1e-9]*/3e-3/*-choose*/;

        var bad = new System.Text.StringBuilder();
        for (int k = 0; k < taus.Length; k++)
        {
            BuildSparseDesign(m, n, out var A, out var b, 4242u);
            var As = ToBsr1x1(in A, Allocator.Temp);

            var xd = new fProxyN(n, Allocator.Temp);
            var xs = new fProxyN(n, Allocator.Temp);
            LP.ladFN(in A, in b, taus[k], ref xd, out double objD);
            LP.ladFN(in As, in b, taus[k], ref xs, out double objS);

            if (System.Math.Abs(objS - objD) > relTol * (1.0 + objD))
                bad.AppendLine($"  tau={taus[k]}: sparse L1 {objS} vs dense {objD}");

            A.Dispose(); b.Dispose(); As.Dispose(); xd.Dispose(); xs.Dispose();
        }
        Assert.That(bad.ToString(), Is.Empty, "sparse quantile fit disagrees with dense:\n" + bad);
    }

    // A design whose rows have WILDLY different nonzero counts, so the block-row walk in the sparse
    // weighted Gram is exercised with short and long rows in the same matrix. A Gram that mishandled
    // multi-entry rows (double-counting, or skipping unsorted column pairs) shows up here.
    [Test]
    public void RaggedRowPatternMatchesDense()
    {
        int m = 96, n = 8;
        var A = new fProxyMxN(m, n, Allocator.Temp);
        var b = new fProxyN(m, Allocator.Temp);
        var rng = new Unity.Mathematics.Random(777u);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++) A[i, j] = (fProxy)0;
            int cnt = 1 + (i % n);                       // 1..n nonzeros, cycling
            for (int t = 0; t < cnt; t++)
                A[i, (i * 3 + t * 5) % n] = rng.NextFProxy(0.5f, 2f);
            double acc = 0;
            for (int j = 0; j < n; j++) acc += (double)A[i, j] * (1.0 + 0.25 * j);
            b[i] = (fProxy)(acc + rng.NextFProxy(-0.2f, 0.2f));
        }
        b[5] = (fProxy)((float)b[5] + 30f);
        b[61] = (fProxy)((float)b[61] - 30f);

        var As = ToBsr1x1(in A, Allocator.Temp);
        var xd = new fProxyN(n, Allocator.Temp);
        var xs = new fProxyN(n, Allocator.Temp);
        LP.ladFN(in A, in b, ref xd, out double objD);
        LP.ladFN(in As, in b, ref xs, out double objS);

        Assert.That(objS, Is.EqualTo(objD).Within(/*+choose[3e-3|1e-9]*/3e-3/*-choose*/ * (1.0 + objD)),
            $"ragged-pattern sparse L1 {objS} vs dense {objD}");

        A.Dispose(); b.Dispose(); As.Dispose(); xd.Dispose(); xs.Dispose();
    }
}
