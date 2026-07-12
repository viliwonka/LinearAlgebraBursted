using System.Diagnostics;
using LinearAlgebra;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Loadout knapsack: 16 gear items (weapons/armor/utility) with weight, value and
    /// energy cost, a max-weapons category cap, and one mandatory item. Builds a binary
    /// MIP (maximize value == minimize -value) and solves it on demand (button press,
    /// not per frame) via <see cref="MIP.solve"/> inside a Burst job.
    /// </summary>
    public class LoadoutMIPDemo : MonoBehaviour
    {
        public enum LoadoutCategory : byte { Weapon = 0, Armor = 1, Utility = 2 }

        const int ItemCount = 16;

        static readonly string[] ItemNames =
        {
            "Combat Knife", "Sidearm Pistol", "Assault Rifle", "Sniper Rifle", "Shotgun", "Rocket Launcher",
            "Light Vest", "Kevlar Plates", "Powered Exosuit", "Riot Shield",
            "Med Kit", "Frag Grenades", "Smoke Grenades", "Grappling Hook", "Portable Shield Generator", "Cloaking Device",
        };

        static readonly LoadoutCategory[] Categories =
        {
            LoadoutCategory.Weapon, LoadoutCategory.Weapon, LoadoutCategory.Weapon,
            LoadoutCategory.Weapon, LoadoutCategory.Weapon, LoadoutCategory.Weapon,
            LoadoutCategory.Armor, LoadoutCategory.Armor, LoadoutCategory.Armor, LoadoutCategory.Armor,
            LoadoutCategory.Utility, LoadoutCategory.Utility, LoadoutCategory.Utility,
            LoadoutCategory.Utility, LoadoutCategory.Utility, LoadoutCategory.Utility,
        };

        // Per-item weight/value/energy cost -- inspector-tunable, index-aligned with ItemNames/Categories.
        public float[] weight = { 1f, 2f, 6f, 8f, 7f, 12f, 3f, 5f, 9f, 6f, 2f, 1f, 1f, 2f, 4f, 3f };
        public float[] value = { 3f, 6f, 14f, 18f, 12f, 22f, 8f, 12f, 20f, 9f, 7f, 5f, 3f, 6f, 10f, 15f };
        public float[] energy = { 0f, 1f, 3f, 4f, 2f, 6f, 0f, 0f, 5f, 0f, 0f, 0f, 0f, 1f, 4f, 5f };

        [Range(1f, 40f)] public float weightCapacity = 20f;
        [Range(1f, 20f)] public float energyBudget = 10f;
        [Range(0, 6)] public int maxWeapons = 2;
        [Range(0, ItemCount - 1)] public int mandatoryItemIndex = 0;

        [Range(0, 500)] public int maxNodes = 0;     // 0 = unlimited
        [Range(0f, 0.2f)] public float relGap = 0f;  // 0 = exact

        bool hasResult;
        MIPStatus lastStatus;
        bool hasIncumbent;
        double lastValue, lastValueBound, lastGap;
        int lastNodes, lastIterations;
        float lastSolveMs;
        readonly bool[] chosen = new bool[ItemCount];
        float chosenWeight, chosenEnergy;
        int chosenWeapons;
        string statusMessage = "";

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 480, 620), GUI.skin.box);
            GUILayout.Label("Loadout MIP optimizer -- binary knapsack + energy/weapon-cap/mandatory-item constraints");

            weightCapacity = LabeledSlider($"Weight cap {weightCapacity:F0}", weightCapacity, 1f, 40f);
            energyBudget = LabeledSlider($"Energy budget {energyBudget:F0}", energyBudget, 1f, 20f);
            maxWeapons = (int)LabeledSlider($"Max weapons {maxWeapons}", maxWeapons, 0, 6.49f);
            int mi = Mathf.Clamp(mandatoryItemIndex, 0, ItemCount - 1);
            mandatoryItemIndex = Mathf.Clamp((int)LabeledSlider($"Mandatory: {ItemNames[mi]}", mandatoryItemIndex, 0, ItemCount - 1 + 0.49f), 0, ItemCount - 1);
            maxNodes = (int)LabeledSlider($"Max nodes {(maxNodes == 0 ? "unlimited" : maxNodes.ToString())}", maxNodes, 0, 500);
            relGap = LabeledSlider($"Rel gap {relGap:P1}", relGap, 0f, 0.2f);

            if (GUILayout.Button("Solve Loadout")) Solve();

            GUILayout.Space(6);

            if (!hasResult)
            {
                GUILayout.Label("Press Solve to run the optimizer.");
            }
            else if (lastStatus == MIPStatus.Infeasible)
            {
                GUILayout.Label($"INFEASIBLE ({lastSolveMs:F2} ms) -- {statusMessage}");
            }
            else if (lastStatus == MIPStatus.Unbounded)
            {
                GUILayout.Label("UNBOUNDED -- unexpected for a bounded binary MIP; check weight/xu values.");
            }
            else if (!hasIncumbent)
            {
                string statusText = lastStatus == MIPStatus.GapLimit ? "GAP LIMIT" : lastStatus == MIPStatus.NodeLimit ? "NODE LIMIT" : "MAX ITERATIONS";
                GUILayout.Label($"{statusText} -- no feasible incumbent found yet within the budget ({lastSolveMs:F2} ms, nodes={lastNodes}, lpIter={lastIterations}). Raise maxNodes/relGap.");
            }
            else
            {
                string statusText = lastStatus == MIPStatus.Optimal ? "OPTIMAL"
                                   : lastStatus == MIPStatus.GapLimit ? "GAP LIMIT (incumbent shown)"
                                   : lastStatus == MIPStatus.NodeLimit ? "NODE LIMIT (incumbent shown)"
                                   : "MAX ITERATIONS (incumbent shown)";
                GUILayout.Label($"{statusText} -- {lastSolveMs:F2} ms, nodes={lastNodes}, lpIter={lastIterations}");
                GUILayout.Label($"value={lastValue:F1}  proven bound<={lastValueBound:F1}  gap={lastGap:P2}");
                GUILayout.Label($"weight={chosenWeight:F1}/{weightCapacity:F0}   energy={chosenEnergy:F1}/{energyBudget:F0}   weapons={chosenWeapons}/{maxWeapons}");

                GUILayout.Space(4);
                for (int j = 0; j < ItemCount; j++)
                {
                    string mark = chosen[j] ? "[x]" : "[ ]";
                    string tag = j == mi ? " (mandatory)" : "";
                    GUILayout.Label($"{mark} {ItemNames[j]}{tag} -- w{weight[j]:F0} v{value[j]:F0} e{energy[j]:F0} ({Categories[j]})");
                }
            }

            GUILayout.EndArea();
        }

        /// <summary>Builds the binary MIP from the current item table/caps and solves it once. Allocates
        /// and disposes every native container within this call -- no per-frame or persistent state.</summary>
        void Solve()
        {
            int mi = Mathf.Clamp(mandatoryItemIndex, 0, ItemCount - 1);
            const int m = 4;   // weight, energy, weapon-count cap, mandatory-item

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
                A[0, j] = weight[j];
                A[1, j] = energy[j];
                A[2, j] = Categories[j] == LoadoutCategory.Weapon ? 1f : 0f;
                A[3, j] = j == mi ? 1f : 0f;
                c[j] = -value[j];   // maximize value == minimize -value
                xl[j] = 0f; xu[j] = 1f;
                integrality[j] = 1;
            }
            b[0] = weightCapacity; b[1] = energyBudget; b[2] = maxWeapons; b[3] = 1f;
            senses[0] = ConstraintSense.LessEqual;
            senses[1] = ConstraintSense.LessEqual;
            senses[2] = ConstraintSense.LessEqual;
            senses[3] = ConstraintSense.GreaterEqual;

            var job = new LoadoutMIPJob
            {
                A = A, B = b, C = c, Xl = xl, Xu = xu, X = x,
                Senses = senses, Integrality = integrality,
                MaxNodes = maxNodes, RelGap = relGap,
                Out = outStats,
            };

            var sw = Stopwatch.StartNew();
            job.Run();
            sw.Stop();
            lastSolveMs = (float)sw.Elapsed.TotalMilliseconds;

            lastStatus = (MIPStatus)(int)outStats[5];
            hasIncumbent = lastStatus != MIPStatus.Infeasible && lastStatus != MIPStatus.Unbounded && !double.IsPositiveInfinity(outStats[0]);
            lastValue = -outStats[0];
            lastValueBound = -outStats[1];
            lastGap = outStats[2];
            lastNodes = (int)outStats[3];
            lastIterations = (int)outStats[4];

            chosenWeight = 0f; chosenEnergy = 0f; chosenWeapons = 0;
            for (int j = 0; j < ItemCount; j++)
            {
                chosen[j] = x[j] > 0.5f;
                if (chosen[j])
                {
                    chosenWeight += weight[j];
                    chosenEnergy += energy[j];
                    if (Categories[j] == LoadoutCategory.Weapon) chosenWeapons++;
                }
            }

            if (lastStatus == MIPStatus.Infeasible)
            {
                string reason = "";
                if (weight[mi] > weightCapacity) reason += "weight cap is below the mandatory item's own weight. ";
                if (energy[mi] > energyBudget) reason += "energy budget is below the mandatory item's own energy cost. ";
                if (Categories[mi] == LoadoutCategory.Weapon && maxWeapons < 1) reason += "max-weapons cap is 0 but the mandatory item is a weapon. ";
                statusMessage = reason.Length > 0 ? reason
                    : "combined weight/energy/weapon caps leave no feasible loadout containing the mandatory item.";
            }

            hasResult = true;

            A.Dispose(); b.Dispose(); c.Dispose(); xl.Dispose(); xu.Dispose(); x.Dispose();
            senses.Dispose(); integrality.Dispose(); outStats.Dispose();
        }

        static float LabeledSlider(string label, float v, float lo, float hi)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(230));
            v = GUILayout.HorizontalSlider(v, lo, hi, GUILayout.Width(210));
            GUILayout.EndHorizontal();
            return v;
        }
    }

    /// <summary>One on-demand loadout MIP solve. No warm state is carried between calls.</summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct LoadoutMIPJob : IJob
    {
        public floatMxN A;
        public floatN B, C, Xl, Xu, X;
        [ReadOnly] public NativeArray<ConstraintSense> Senses;
        [ReadOnly] public NativeArray<byte> Integrality;
        public int MaxNodes;
        public double RelGap;
        public NativeArray<double> Out;

        public void Execute()
        {
            MIPInfo info = MIP.solve(in A, in B, in C, in Senses, in Xl, in Xu, in Integrality, ref X, out double objective,
                                     MaxNodes, 0, 0.0, RelGap);
            Out[0] = objective;
            Out[1] = info.dualBound;
            Out[2] = info.gap;
            Out[3] = info.nodes;
            Out[4] = info.lpIterations;
            Out[5] = (double)(int)info.status;
        }
    }
}
