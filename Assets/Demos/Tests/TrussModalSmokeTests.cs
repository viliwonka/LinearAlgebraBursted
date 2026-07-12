using LinearAlgebra;
using LinearAlgebra.Sparse;
using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// Headless smoke test for <see cref="TrussModalDemo"/>: assembles a small braced-square
    /// truss (same node/member/penalty-BC assembly pattern as
    /// <see cref="DemoSmokeTests.TrussEigenJob_BracedSquare_PositiveSpectrum"/>) plus a lumped
    /// diagonal mass matrix, runs <see cref="TrussModalJob"/> once, and checks the generalized
    /// eigenpairs it returns rather than trusting the solver's own self-reported residual.
    /// </summary>
    public class TrussModalSmokeTests
    {
        [Test]
        public void TrussModalJob_BracedSquare_ConvergesToAscendingPositiveSpectrum()
        {
            // 4-node square truss, both diagonals braced, two nodes pinned -- fully determinate.
            float2[] nodes = { new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1) };
            int2[] bars = {
                new int2(0, 1), new int2(1, 2), new int2(2, 3), new int2(3, 0),
                new int2(0, 2), new int2(1, 3),
            };
            // K=2 mirrors the exact modeCount already proven safe for this geometry by
            // DemoSmokeTests.TrussEigenJob_BracedSquare_PositiveSpectrum (same 4-node square,
            // same pinned dof, same penalty magnitude) -- the pinned dof's penalty-shifted
            // frequencies sit well above the two lowest structural modes checked here.
            const int modeCount = 2;
            const float nodeMass = 1f;
            int n = nodes.Length * 2;

            var arena = new Arena(Allocator.Temp);

            var kBuilder = new floatBSRBuilder(nodes.Length, nodes.Length, 2, 2, Allocator.Temp, 32);
            var mBuilder = new floatBSRBuilder(nodes.Length, nodes.Length, 2, 2, Allocator.Temp, nodes.Length);
            foreach (var bar in bars)
            {
                float2 d = nodes[bar.y] - nodes[bar.x];
                float2 u = math.normalize(d);
                float k = 5f / math.length(d);
                int lo = math.min(bar.x, bar.y), hi = math.max(bar.x, bar.y);
                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 2; c++)
                    {
                        float v = k * u[r] * u[c];
                        kBuilder.AddValue(2 * bar.x + r, 2 * bar.x + c, v);
                        kBuilder.AddValue(2 * bar.y + r, 2 * bar.y + c, v);
                        kBuilder.AddValue(2 * hi + r, 2 * lo + c, -v);
                    }
            }
            // penalty within ~3 decades of bar stiffness -- see TrussStabilityDemo.Build
            for (int d = 0; d < 2; d++)
            {
                kBuilder.AddValue(0 + d, 0 + d, 1e3f);
                kBuilder.AddValue(2 + d, 2 + d, 1e3f);
            }
            for (int i = 0; i < nodes.Length; i++)
                for (int d = 0; d < 2; d++)
                    mBuilder.AddValue(2 * i + d, 2 * i + d, nodeMass);

            var A = kBuilder.ToBSRSymmetric(ref arena);
            kBuilder.Dispose();
            var B = mBuilder.ToBSRSymmetric(ref arena);
            mBuilder.Dispose();

            var precond = arena.floatBlockJacobi(in A);
            var cache = arena.floatLOBPCGCache(n, modeCount);
            var outStats = new NativeArray<float>(3, Allocator.TempJob);

            var job = new TrussModalJob
            {
                A = new floatBSROperator(in A), B = new floatBSROperator(in B),
                Precond = precond, Cache = cache, Out = outStats, K = modeCount,
            };
            IJobExtensions.RunByRef(ref job);

            Assert.IsTrue(outStats[1] == 1f,
                $"LOBPCG did not converge (iterations={outStats[0]}, maxResidual={outStats[2]:E2})");

            var lambda = job.Cache.lambda;
            var modes = job.Cache.X;

            // (c) frequencies positive and ascending
            for (int i = 0; i < modeCount; i++)
                Assert.IsTrue(lambda[i] > 0f, $"lambda[{i}] = {lambda[i]} is not positive");
            for (int i = 1; i < modeCount; i++)
                Assert.IsTrue(lambda[i - 1] <= lambda[i] + 1e-4f,
                    $"eigenvalues not ascending: lambda[{i - 1}]={lambda[i - 1]} > lambda[{i}]={lambda[i]}");

            // (b) per-pair residual ||A*phi - lambda*B*phi|| small relative to ||A*phi||,
            // recomputed independently via spMV rather than trusting info.maxResidual.
            var phi = new floatN(n, Allocator.Temp);
            var Aphi = new floatN(n, Allocator.Temp);
            var Bphi = new floatN(n, Allocator.Temp);
            for (int i = 0; i < modeCount; i++)
            {
                for (int c = 0; c < n; c++) phi[c] = modes[i, c];
                BSR.spMV(in A, in phi, ref Aphi);
                BSR.spMV(in B, in phi, ref Bphi);

                float resNorm2 = 0f, aNorm2 = 0f;
                for (int c = 0; c < n; c++)
                {
                    float r = Aphi[c] - lambda[i] * Bphi[c];
                    resNorm2 += r * r;
                    aNorm2 += Aphi[c] * Aphi[c];
                }
                float resNorm = math.sqrt(resNorm2);
                float aNorm = math.sqrt(aNorm2);
                Assert.IsTrue(resNorm <= 1e-2f * math.max(aNorm, 1f),
                    $"mode {i}: residual {resNorm} too large relative to ||A*phi|| = {aNorm}");
            }

            // (d) no native leaks
            outStats.Dispose();
            arena.Dispose();
        }
    }
}
