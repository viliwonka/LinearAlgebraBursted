using BULA;
using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LinearAlgebraDemos.Tests
{
    /// <summary>
    /// Headless smoke test for the loadout MIP demo: rebuilds its exact default item
    /// table and constraint rows, runs <see cref="LoadoutMIPJob"/>, and cross-checks the
    /// result against a brute-force enumeration over all 2^16 item subsets.
    /// </summary>
    public class LoadoutSmokeTests
    {
        // SAME literal data as LoadoutMIPDemo's default fields (ItemNames/Categories/weight/value/energy)
        // and default caps (weightCapacity/energyBudget/maxWeapons/mandatoryItemIndex).
        const int ItemCount = 16;

        static readonly float[] Weight = { 1f, 2f, 6f, 8f, 7f, 12f, 3f, 5f, 9f, 6f, 2f, 1f, 1f, 2f, 4f, 3f };
        static readonly float[] Value = { 3f, 6f, 14f, 18f, 12f, 22f, 8f, 12f, 20f, 9f, 7f, 5f, 3f, 6f, 10f, 15f };
        static readonly float[] Energy = { 0f, 1f, 3f, 4f, 2f, 6f, 0f, 0f, 5f, 0f, 0f, 0f, 0f, 1f, 4f, 5f };
        static readonly bool[] IsWeapon = { true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false };
        const int MandatoryIndex = 0;

        const float WeightCapacity = 20f;
        const float EnergyBudget = 10f;
        const int MaxWeapons = 2;

        [Test]
        public void LoadoutMIPJob_DefaultTable_MatchesBruteForce()
        {
            const int m = 4;
            var A = new floatMxN(m, ItemCount, Allocator.TempJob);
            var b = new floatN(m, Allocator.TempJob);
            var c = new floatN(ItemCount, Allocator.TempJob);
            var xl = new floatN(ItemCount, Allocator.TempJob);
            var xu = new floatN(ItemCount, Allocator.TempJob);
            var x = new floatN(ItemCount, Allocator.TempJob);
            var senses = new NativeArray<ConstraintSense>(m, Allocator.TempJob);
            var integrality = new NativeArray<byte>(ItemCount, Allocator.TempJob);
            var outStats = new NativeArray<double>(6, Allocator.TempJob);

            for (int j = 0; j < ItemCount; j++)
            {
                A[0, j] = Weight[j];
                A[1, j] = Energy[j];
                A[2, j] = IsWeapon[j] ? 1f : 0f;
                A[3, j] = j == MandatoryIndex ? 1f : 0f;
                c[j] = -Value[j];
                xl[j] = 0f; xu[j] = 1f;
                integrality[j] = 1;
            }
            b[0] = WeightCapacity; b[1] = EnergyBudget; b[2] = MaxWeapons; b[3] = 1f;
            senses[0] = ConstraintSense.LessEqual;
            senses[1] = ConstraintSense.LessEqual;
            senses[2] = ConstraintSense.LessEqual;
            senses[3] = ConstraintSense.GreaterEqual;

            var job = new LoadoutMIPJob
            {
                A = A, B = b, C = c, Xl = xl, Xu = xu, X = x,
                Senses = senses, Integrality = integrality,
                MaxNodes = 0, RelGap = 0.0,
                Out = outStats,
            };
            job.Run();

            var status = (MIPStatus)(int)outStats[5];
            Assert.IsTrue(status == MIPStatus.Optimal, $"expected Optimal, got {status}");

            double solverValue = -outStats[0];

            float wSum = 0f, eSum = 0f; int weapons = 0;
            var chosen = new bool[ItemCount];
            for (int j = 0; j < ItemCount; j++)
            {
                chosen[j] = x[j] > 0.5f;
                if (chosen[j])
                {
                    wSum += Weight[j]; eSum += Energy[j];
                    if (IsWeapon[j]) weapons++;
                }
            }
            Assert.IsTrue(chosen[MandatoryIndex], "mandatory item not selected");
            Assert.IsTrue(wSum <= WeightCapacity + 1e-3f, $"weight cap violated: {wSum}");
            Assert.IsTrue(eSum <= EnergyBudget + 1e-3f, $"energy budget violated: {eSum}");
            Assert.IsTrue(weapons <= MaxWeapons, $"weapon cap violated: {weapons}");

            // Brute force over all 2^16 subsets containing the mandatory item -- the table is small
            // enough to enumerate exhaustively as an independent check on the B&B result.
            double bestValue = double.NegativeInfinity;
            for (int mask = 0; mask < (1 << ItemCount); mask++)
            {
                if ((mask & (1 << MandatoryIndex)) == 0) continue;
                float w = 0f, e = 0f; int wc = 0; double v = 0;
                for (int j = 0; j < ItemCount; j++)
                {
                    if ((mask & (1 << j)) == 0) continue;
                    w += Weight[j]; e += Energy[j]; v += Value[j];
                    if (IsWeapon[j]) wc++;
                }
                if (w <= WeightCapacity && e <= EnergyBudget && wc <= MaxWeapons)
                    bestValue = math.max(bestValue, v);
            }

            Assert.IsTrue(bestValue > double.NegativeInfinity, "brute force found no feasible subset (test setup bug)");
            Assert.IsTrue(math.abs(solverValue - bestValue) < 1e-3, $"solver value {solverValue} != brute-force optimum {bestValue}");

            A.Dispose(); b.Dispose(); c.Dispose(); xl.Dispose(); xu.Dispose(); x.Dispose();
            senses.Dispose(); integrality.Dispose(); outStats.Dispose();
        }
    }
}
