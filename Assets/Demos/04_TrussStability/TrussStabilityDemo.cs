using System.Diagnostics;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Structural stability of a small house frame. The truss stiffness matrix is
    /// assembled into 2×2-block symmetric BSR (lower-block storage); a Burst job
    /// runs block-Jacobi-preconditioned LOBPCG for the 4 smallest eigenpairs every
    /// frame (warm-started from the previous frame's cache). Toggle bracing
    /// members off and watch lambda1 collapse toward a mechanism; the
    /// corresponding mode shape is animated on the frame.
    /// </summary>
    public class TrussStabilityDemo : MonoBehaviour
    {
        [Range(0.5f, 20f)] public float stiffnessEA = 8f;
        [Range(0f, 0.5f)] public float modeAmplitude = 0.15f;
        [Range(0, 3)] public int shownMode;

        // house frame: 4 bottom, 4 mid, 1 ridge
        static readonly float2[] Nodes =
        {
            new float2(0, 0), new float2(1, 0), new float2(2, 0), new float2(3, 0),
            new float2(0, 1), new float2(1, 1), new float2(2, 1), new float2(3, 1),
            new float2(1.5f, 1.9f),
        };
        // fixed members: chords, columns, rafters
        static readonly int2[] Fixed =
        {
            new int2(0, 1), new int2(1, 2), new int2(2, 3),      // bottom chord
            new int2(4, 5), new int2(5, 6), new int2(6, 7),      // mid chord
            new int2(0, 4), new int2(1, 5), new int2(2, 6), new int2(3, 7),   // columns
            new int2(4, 8), new int2(5, 8), new int2(6, 8), new int2(7, 8),   // roof
        };
        // toggleable diagonal braces
        static readonly int2[] Braces =
        {
            new int2(0, 5), new int2(1, 6), new int2(2, 7),
            new int2(1, 4), new int2(2, 5), new int2(3, 6),
        };
        public bool[] braceOn = { true, true, true, true, true, true };

        const int K = 4;
        int N => Nodes.Length * 2;

        Arena arena;
        floatBSR A;
        floatBlockJacobi precond;
        floatLOBPCGCache cache;
        floatN lambda;      // arena-owned view of cache.lambda after solve
        floatMxN modes;     // arena-owned view of cache.X (k × n)
        bool built;
        float builtEA;
        bool[] builtBraces;
        NativeArray<float> outStats;   // [0] iterations, [1] converged
        float frameMs;

        void OnEnable()
        {
            outStats = new NativeArray<float>(2, Allocator.Persistent);
            builtBraces = (bool[])braceOn.Clone();
            Build();
        }

        void OnDisable()
        {
            if (built) { arena.Dispose(); built = false; }
            if (outStats.IsCreated) outStats.Dispose();
        }

        void Build()
        {
            if (built) arena.Dispose();
            arena = new Arena(Allocator.Persistent);

            int nb = Nodes.Length;
            var builder = new floatBSRBuilder(nb, nb, 2, 2, Allocator.Temp, 64);

            void AddBar(int a, int b)
            {
                float2 d = Nodes[b] - Nodes[a];
                float L = math.length(d);
                float2 u = d / L;
                float k = stiffnessEA / L;
                // 2×2 block k·uuT on both diagonals, -k·uuT at the LOWER (max,min) position
                int lo = math.min(a, b), hi = math.max(a, b);
                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 2; c++)
                    {
                        float v = k * u[r] * u[c];
                        builder.AddValue(2 * a + r, 2 * a + c, v);
                        builder.AddValue(2 * b + r, 2 * b + c, v);
                        builder.AddValue(2 * hi + r, 2 * lo + c, -v);
                    }
            }

            foreach (var m in Fixed) AddBar(m.x, m.y);
            for (int i = 0; i < Braces.Length; i++)
                if (braceOn[i]) AddBar(Braces[i].x, Braces[i].y);

            // pinned supports at nodes 0 and 3: penalty on their diagonal blocks.
            // Keep the penalty within ~3 decades of the bar stiffness: float LOBPCG
            // forms Gram matrices ~penalty², and a 1e6 penalty pushes the O(EA)
            // eigenvalues below the float noise floor (they come back as exactly 0).
            for (int d = 0; d < 2; d++)
            {
                builder.AddValue(0 + d, 0 + d, 1e3f);
                builder.AddValue(6 + d, 6 + d, 1e3f);   // node 3 → dof 6,7
            }

            A = builder.ToBSRSymmetric(ref arena);
            builder.Dispose();
            precond = arena.floatBlockJacobi(in A);
            cache = arena.floatLOBPCGCache(N, K);

            built = true;
            builtEA = stiffnessEA;
            braceOn.CopyTo(builtBraces, 0);
        }

        void Update()
        {
            bool dirty = builtEA != stiffnessEA;
            for (int i = 0; i < braceOn.Length; i++) dirty |= braceOn[i] != builtBraces[i];
            if (dirty) Build();

            var job = new TrussEigenJob
            {
                Op = new floatBSROperator(in A),
                Precond = precond,
                Cache = cache,
                Out = outStats,
                K = K,
            };

            var sw = Stopwatch.StartNew();
            IJobExtensions.RunByRef(ref job);
            sw.Stop();
            frameMs = (float)sw.Elapsed.TotalMilliseconds;

            cache = job.Cache;
            lambda = cache.lambda;   // default until the first Update -> lambda.IsCreated gates readers
            modes = cache.X;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !built || !lambda.IsCreated) return;

            float wob = modeAmplitude * math.sin(6f * Time.time);
            Vector3 P(int node)
            {
                float2 p = Nodes[node];
                float dx = modes[shownMode, 2 * node] * wob;
                float dy = modes[shownMode, 2 * node + 1] * wob;
                return new Vector3(p.x + dx, p.y + dy, 0f);
            }

            Gizmos.color = Color.white;
            foreach (var m in Fixed) Gizmos.DrawLine(P(m.x), P(m.y));
            Gizmos.color = Color.yellow;
            for (int i = 0; i < Braces.Length; i++)
                if (braceOn[i]) Gizmos.DrawLine(P(Braces[i].x), P(Braces[i].y));
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(P(0), 0.06f); Gizmos.DrawSphere(P(3), 0.06f);
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 420, 260), GUI.skin.box);
            GUILayout.Label($"Truss stability — LOBPCG k={K} over {N}-dof BSR, {frameMs:F2} ms/frame");
            GUILayout.Label($"iters: {outStats[0]:F0} (warm)   converged: {outStats[1] == 1f}");
            if (built && lambda.IsCreated)
            {
                GUILayout.Label($"lambda = [{lambda[0]:F3}, {lambda[1]:F3}, {lambda[2]:F3}, {lambda[3]:F3}]");
                bool unstable = lambda[0] < 0.05f * stiffnessEA;
                var style = new GUIStyle(GUI.skin.label);
                style.normal.textColor = unstable ? Color.red : Color.green;
                GUILayout.Label(unstable
                    ? "lambda1 ≈ 0 — near-mechanism, structure is UNSTABLE"
                    : "structure is stiff (no soft modes)", style);
            }

            GUILayout.Label("Diagonal braces:");
            GUILayout.BeginHorizontal();
            for (int i = 0; i < Braces.Length; i++)
                braceOn[i] = GUILayout.Toggle(braceOn[i], $"{Braces[i].x}-{Braces[i].y}");
            GUILayout.EndHorizontal();
            shownMode = (int)LabeledSlider($"mode {shownMode} (lambda={((built && lambda.IsCreated) ? lambda[shownMode] : 0f):F3})", shownMode, 0, 3.49f);
            stiffnessEA = LabeledSlider($"EA {stiffnessEA:F1}", stiffnessEA, 0.5f, 20f);
            GUILayout.EndArea();
        }

        static float LabeledSlider(string label, float v, float lo, float hi)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(150));
            v = GUILayout.HorizontalSlider(v, lo, hi, GUILayout.Width(220));
            GUILayout.EndHorizontal();
            return v;
        }
    }

    /// <summary>Warm LOBPCG smallest-k eigenpairs of the truss stiffness matrix.</summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct TrussEigenJob : IJob
    {
        [ReadOnly] public floatBSROperator Op;
        [ReadOnly] public floatBlockJacobi Precond;
        public floatLOBPCGCache Cache;
        public NativeArray<float> Out;
        public int K;

        public void Execute()
        {
            LOBPCGInfo info = Eigen.lobpcg(in Op, in Precond, ref Cache, K, 1e-4f, 200);
            Out[0] = info.iterations;
            Out[1] = info ? 1f : 0f;
        }
    }
}
