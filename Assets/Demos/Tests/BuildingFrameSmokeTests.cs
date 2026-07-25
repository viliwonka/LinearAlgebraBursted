using System.Collections.Generic;
using BULA;
using BULA.Sparse;
using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// Headless smoke test for <see cref="BuildingFrameStabilityDemo"/>: assembles a small
    /// fully-braced 2×2-bay × 2-story frame (3×3-node grid × 3 levels, 3×3-block BSR, same
    /// column/beam/perimeter-brace/penalty-BC assembly pattern as the demo), runs
    /// <see cref="TrussEigenJobIC0"/> (IC(0)-preconditioned LOBPCG) once, and checks the eigenpairs
    /// it returns rather than trusting the solver's own self-reported residual.
    /// </summary>
    public class BuildingFrameSmokeTests
    {
        static int NodeIdx(int i, int j, int l, int nw, int nd) => (l * nd + j) * nw + i;

        [Test]
        public void BuildingFrame_BracedFrame_ConvergesToAscendingPositiveSpectrum()
        {
            const int baysX = 2, baysZ = 2, stories = 2;
            int nw = baysX + 1, nd = baysZ + 1, levels = stories + 1;
            const float w = 1f, dd = 1f, h = 1f;

            var nodes = new float3[nw * nd * levels];
            for (int l = 0; l < levels; l++)
                for (int j = 0; j < nd; j++)
                    for (int i = 0; i < nw; i++)
                        nodes[NodeIdx(i, j, l, nw, nd)] = new float3(i * w, l * h, j * dd);

            var bars = new List<int2>();
            for (int l = 0; l < stories; l++)                      // columns
                for (int j = 0; j < nd; j++)
                    for (int i = 0; i < nw; i++)
                        bars.Add(new int2(NodeIdx(i, j, l, nw, nd), NodeIdx(i, j, l + 1, nw, nd)));
            for (int l = 1; l < levels; l++)                       // two-way floor beams
            {
                for (int j = 0; j < nd; j++)
                    for (int i = 0; i < baysX; i++)
                        bars.Add(new int2(NodeIdx(i, j, l, nw, nd), NodeIdx(i + 1, j, l, nw, nd)));
                for (int j = 0; j < baysZ; j++)
                    for (int i = 0; i < nw; i++)
                        bars.Add(new int2(NodeIdx(i, j, l, nw, nd), NodeIdx(i, j + 1, l, nw, nd)));
            }
            for (int l = 1; l < levels; l++)                       // rigid-diaphragm floor bracing
                for (int j = 0; j < baysZ; j++)
                    for (int i = 0; i < baysX; i++)
                        bars.Add(new int2(NodeIdx(i, j, l, nw, nd), NodeIdx(i + 1, j + 1, l, nw, nd)));
            for (int s = 0; s < stories; s++)                      // perimeter bracing (all on)
            {
                for (int i = 0; i < baysX; i++)
                {
                    bars.Add(new int2(NodeIdx(i, 0, s, nw, nd), NodeIdx(i + 1, 0, s + 1, nw, nd)));
                    bars.Add(new int2(NodeIdx(i, baysZ, s, nw, nd), NodeIdx(i + 1, baysZ, s + 1, nw, nd)));
                }
                for (int j = 0; j < baysZ; j++)
                {
                    bars.Add(new int2(NodeIdx(0, j, s, nw, nd), NodeIdx(0, j + 1, s + 1, nw, nd)));
                    bars.Add(new int2(NodeIdx(baysX, j, s, nw, nd), NodeIdx(baysX, j + 1, s + 1, nw, nd)));
                }
            }

            const int K = 3;
            int n = nodes.Length * 3;

            var builder = new floatBSRBuilder(nodes.Length, nodes.Length, 3, 3, Allocator.Temp, bars.Count * 27);

            foreach (var bar in bars)
            {
                float3 d = nodes[bar.y] - nodes[bar.x];
                float3 u = math.normalize(d);
                float k = 5f / math.length(d);
                int lo = math.min(bar.x, bar.y), hi = math.max(bar.x, bar.y);
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                    {
                        float v = k * u[r] * u[c];
                        builder.AddValue(3 * bar.x + r, 3 * bar.x + c, v);
                        builder.AddValue(3 * bar.y + r, 3 * bar.y + c, v);
                        builder.AddValue(3 * hi + r, 3 * lo + c, -v);
                    }
            }
            for (int j = 0; j < nd; j++)                           // pin every ground node
                for (int i = 0; i < nw; i++)
                {
                    int node = NodeIdx(i, j, 0, nw, nd);
                    for (int ddof = 0; ddof < 3; ddof++)
                        builder.AddValue(3 * node + ddof, 3 * node + ddof, 1e3f);
                }

            var A = builder.ToBSRSymmetric(Allocator.Temp);
            builder.Dispose();

            var precond = new floatIC0(in A, Allocator.Temp);
            // guard vectors: the doubly-symmetric frame has a near-degenerate soft cluster, so
            // iterate on K+4 vectors and return the K smallest (see BuildingFrameStabilityDemo).
            var cache = new floatLOBPCGCache(n, K + 4, Allocator.Temp);
            var outStats = new NativeArray<float>(2, Allocator.TempJob);

            var job = new TrussEigenJobIC0
            {
                Op = new floatBSROperator(in A), Precond = precond, Cache = cache,
                Out = outStats, K = K,
            };
            IJobExtensions.RunByRef(ref job);

            Assert.IsTrue(outStats[1] == 1f, $"LOBPCG did not converge (iterations={outStats[0]})");

            var lambda = job.Cache.lambda;
            var modes = job.Cache.X;

            for (int i = 0; i < K; i++)
                Assert.IsTrue(lambda[i] > 0f, $"lambda[{i}] = {lambda[i]} is not positive");
            for (int i = 1; i < K; i++)
                Assert.IsTrue(lambda[i - 1] <= lambda[i] + 1e-4f,
                    $"eigenvalues not ascending: lambda[{i - 1}]={lambda[i - 1]} > lambda[{i}]={lambda[i]}");

            // per-pair residual ||A*phi - lambda*phi|| small relative to ||A*phi||, recomputed
            // independently via spMV rather than trusting info.maxResidual.
            var phi = new floatN(n, Allocator.Temp);
            var Aphi = new floatN(n, Allocator.Temp);
            for (int i = 0; i < K; i++)
            {
                for (int c = 0; c < n; c++) phi[c] = modes[i, c];
                BSR.spMV(in A, in phi, ref Aphi);

                float resNorm2 = 0f, aNorm2 = 0f;
                for (int c = 0; c < n; c++)
                {
                    float r = Aphi[c] - lambda[i] * phi[c];
                    resNorm2 += r * r;
                    aNorm2 += Aphi[c] * Aphi[c];
                }
                float resNorm = math.sqrt(resNorm2);
                float aNorm = math.sqrt(aNorm2);
                Assert.IsTrue(resNorm <= 1e-3f * math.max(aNorm, 1f),
                    $"mode {i}: residual {resNorm} too large relative to ||A*phi|| = {aNorm}");
            }

            phi.Dispose(); Aphi.Dispose();
            outStats.Dispose();
        }
    }
}
