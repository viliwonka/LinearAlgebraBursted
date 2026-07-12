using System.Diagnostics;
using LinearAlgebra;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Realtime adjustable linear program: a village economy producing four goods
    /// from three shared resources plus per-good market demand caps. Drag resource
    /// capacities (RHS-only change → warm re-solve, no cache invalidation) or
    /// profits (objective change → cache.matrixVersion bump) and watch the optimal
    /// production mix re-solve every frame via warm-started dual simplex
    /// (LP.solve with LPBasis + floatLPCache) inside a Burst job.
    ///
    /// Interop note: LP.solve mutates scalar fields of LPBasis/floatLPCache
    /// (populated, builtVersion...). A job struct is copied by value, so the job
    /// MUST run via RunByRef and the structs must be copied back afterwards —
    /// with plain Run() the warm state silently degrades to a cold solve every
    /// frame.
    /// </summary>
    public class EconomyLPDemo : MonoBehaviour
    {
        const int Products = 4;    // bread, ale, cheese, pie
        const int Resources = 3;   // grain, water, labor
        const int M = Resources + Products;   // + per-product demand caps

        static readonly string[] ProductNames = { "Bread", "Ale", "Cheese", "Pie" };

        // consumption per unit produced (rows: resources, cols: products)
        static readonly float[,] Use =
        {
            { 2.0f, 1.0f, 0.5f, 1.5f },   // grain
            { 0.5f, 2.0f, 1.0f, 0.5f },   // water
            { 1.0f, 0.5f, 2.0f, 1.5f },   // labor
        };

        [Range(10f, 400f)] public float grainCapacity = 200f;
        [Range(10f, 400f)] public float waterCapacity = 150f;
        [Range(10f, 400f)] public float laborCapacity = 180f;
        public float[] profits = { 3f, 4f, 6f, 5f };
        [Range(5f, 200f)] public float demandCap = 80f;

        floatMxN A;
        floatN b, c, x;
        NativeArray<ConstraintSense> senses;
        NativeArray<float> outStats;   // [0] objective, [1] iterations, [2] ok
        LPBasis basis;
        floatLPCache cache;
        float[] lastProfits = new float[Products];
        float solveMs;
        int coldFrames, warmFrames;
        readonly Stopwatch sw = new Stopwatch();

        void OnEnable()
        {
            A = new floatMxN(M, Products, Allocator.Persistent);
            b = new floatN(M, Allocator.Persistent);
            c = new floatN(Products, Allocator.Persistent);
            x = new floatN(Products, Allocator.Persistent);
            senses = new NativeArray<ConstraintSense>(M, Allocator.Persistent);
            outStats = new NativeArray<float>(3, Allocator.Persistent);
            basis = new LPBasis(Products, M, Allocator.Persistent);
            cache = new floatLPCache(Products, M, Allocator.Persistent);

            for (int r = 0; r < Resources; r++)
                for (int j = 0; j < Products; j++)
                    A[r, j] = Use[r, j];
            for (int j = 0; j < Products; j++)
                A[Resources + j, j] = 1f;   // x_j <= demandCap
            for (int i = 0; i < M; i++)
                senses[i] = ConstraintSense.LessEqual;

            for (int j = 0; j < Products; j++) lastProfits[j] = float.NaN;
        }

        void OnDisable()
        {
            A.Dispose(); b.Dispose(); c.Dispose(); x.Dispose();
            senses.Dispose(); outStats.Dispose();
            basis.Dispose(); cache.Dispose();
        }

        void Update()
        {
            b[0] = grainCapacity; b[1] = waterCapacity; b[2] = laborCapacity;
            for (int j = 0; j < Products; j++) b[Resources + j] = demandCap;

            bool objectiveChanged = false;
            for (int j = 0; j < Products; j++)
            {
                float pj = profits[j];
                if (pj != lastProfits[j]) { objectiveChanged = true; lastProfits[j] = pj; }
                c[j] = -pj;   // LP minimizes; maximize profit = minimize -profit
            }
            if (objectiveChanged)
                cache.matrixVersion++;   // objective is part of the cached computational form

            var job = new EconomyLPJob
            {
                A = A, B = b, C = c, X = x,
                Senses = senses, Basis = basis, Cache = cache, Out = outStats,
            };

            sw.Restart();
            IJobExtensions.RunByRef(ref job);
            sw.Stop();
            solveMs = (float)sw.Elapsed.TotalMilliseconds;

            // reclaim scalar warm state (populated / builtVersion / pricing flags)
            basis = job.Basis;
            cache = job.Cache;

            if (outStats[1] <= 1f) warmFrames++; else coldFrames++;
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 420, 330), GUI.skin.box);
            GUILayout.Label($"Village economy LP — warm dual simplex, {solveMs:F3} ms/frame");
            GUILayout.Label($"profit: {-outStats[0]:F1}   pivots: {outStats[1]:F0}   optimal: {outStats[2] == 1f}   (≤1-pivot frames: {warmFrames}, more: {coldFrames})");

            GUILayout.Space(4);
            grainCapacity = LabeledSlider($"Grain cap {grainCapacity:F0}", grainCapacity, 10f, 400f);
            waterCapacity = LabeledSlider($"Water cap {waterCapacity:F0}", waterCapacity, 10f, 400f);
            laborCapacity = LabeledSlider($"Labor cap {laborCapacity:F0}", laborCapacity, 10f, 400f);
            demandCap = LabeledSlider($"Demand cap {demandCap:F0}", demandCap, 5f, 200f);

            GUILayout.Space(4);
            for (int j = 0; j < Products; j++)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label($"{ProductNames[j]} profit {profits[j]:F1}", GUILayout.Width(130));
                profits[j] = GUILayout.HorizontalSlider(profits[j], 0f, 12f, GUILayout.Width(120));
                float units = x[j];
                GUILayout.Box("", GUILayout.Width(Mathf.Clamp(units, 0f, 140f)), GUILayout.Height(14));
                GUILayout.Label($"{units:F1}");
                GUILayout.EndHorizontal();
            }
            GUILayout.EndArea();
        }

        static float LabeledSlider(string label, float v, float lo, float hi)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(130));
            v = GUILayout.HorizontalSlider(v, lo, hi, GUILayout.Width(240));
            GUILayout.EndHorizontal();
            return v;
        }
    }

    /// <summary>One warm-started LP re-solve. Basis/Cache scalar state must be copied back by the caller.</summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct EconomyLPJob : IJob
    {
        public floatMxN A;
        public floatN B, C, X;
        [ReadOnly] public NativeArray<ConstraintSense> Senses;
        public LPBasis Basis;
        public floatLPCache Cache;
        public NativeArray<float> Out;

        public void Execute()
        {
            LPInfo info = LP.solve(in A, in B, in C, Senses, ref X, out double objective,
                                   ref Basis, ref Cache);
            Out[0] = (float)objective;
            Out[1] = info.iterations;
            Out[2] = info ? 1f : 0f;
        }
    }
}
