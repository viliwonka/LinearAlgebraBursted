using System.Collections.Generic;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// Headless smoke test for <see cref="Truss3DStabilityDemo"/>: assembles a small braced
    /// 2-story tower (3 levels x 4 corners, 3x3-block BSR, same node/member/penalty-BC assembly
    /// pattern as <see cref="Truss3DStabilityDemo.Build"/>), runs <see cref="TrussEigenJobIC0"/>
    /// (IC(0)-preconditioned LOBPCG) once, and checks the eigenpairs it returns rather than
    /// trusting the solver's own self-reported residual.
    /// </summary>
    public class Truss3DSmokeTests
    {
        [Test]
        public void TrussEigenJob_BracedTower3D_ConvergesToAscendingPositiveSpectrum()
        {
            // 2-story square tower, all 4 faces braced at both stories, base pinned: fully
            // determinate 3D space frame (3 dof/node, 3x3 blocks).
            const int stories = 2;
            const int levels = stories + 1;
            const float width = 1f, height = 1f;
            float hw = width * 0.5f;

            float2[] corner = { new float2(-hw, -hw), new float2(hw, -hw), new float2(hw, hw), new float2(-hw, hw) };
            var nodes = new float3[levels * 4];
            for (int l = 0; l < levels; l++)
                for (int c = 0; c < 4; c++)
                    nodes[l * 4 + c] = new float3(corner[c].x, l * height, corner[c].y);

            var bars = new List<int2>();
            for (int l = 0; l < stories; l++)
                for (int c = 0; c < 4; c++)
                    bars.Add(new int2(l * 4 + c, (l + 1) * 4 + c));               // vertical chords
            for (int l = 0; l < levels; l++)
                for (int c = 0; c < 4; c++)
                    bars.Add(new int2(l * 4 + c, l * 4 + (c + 1) % 4));           // horizontal ring beams
            for (int l = 0; l < stories; l++)
                for (int f = 0; f < 4; f++)
                    bars.Add(new int2(l * 4 + f, (l + 1) * 4 + (f + 1) % 4));     // one diagonal per face
            for (int l = 1; l < levels; l++)
                bars.Add(new int2(l * 4 + 0, l * 4 + 2));                          // floor diaphragm brace

            const int K = 3;
            int n = nodes.Length * 3;

            var arena = new Arena(Allocator.Temp);
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
            // penalty within ~3 decades of bar stiffness -- see Truss3DStabilityDemo.Build
            for (int c = 0; c < 4; c++)
                for (int d = 0; d < 3; d++)
                    builder.AddValue(3 * c + d, 3 * c + d, 1e3f);

            var A = builder.ToBSRSymmetric(ref arena);
            builder.Dispose();

            var precond = arena.floatIC0(in A);
            var cache = arena.floatLOBPCGCache(n, K);
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

            // spectrum positive and ascending
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

            // no native leaks
            outStats.Dispose();
            arena.Dispose();
        }
    }
}
