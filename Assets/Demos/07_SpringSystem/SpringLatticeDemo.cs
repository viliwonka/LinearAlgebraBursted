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
    /// Hanging cloth as a mass-spring lattice, integrated implicitly: each frame
    /// solves (M + h²·k·L) v+ = M·v + h·f over 3×3-block sparse BSR with
    /// IC(0)-preconditioned CG (Krylov.cg) inside a Burst job. The system matrix
    /// uses the constant graph-Laplacian approximation, so it is assembled ONCE
    /// (symmetric lower-block storage) and only the right-hand side changes per
    /// frame — the realtime sparse-SPD showcase. Wind slider + poke button.
    /// </summary>
    public class SpringLatticeDemo : MonoBehaviour
    {
        [Range(4, 24)] public int gridWidth = 12;
        [Range(4, 24)] public int gridHeight = 10;
        [Range(10f, 800f)] public float stiffness = 250f;
        [Range(0.01f, 1f)] public float nodeMass = 0.1f;
        [Range(0f, 3f)] public float damping = 0.4f;
        [Range(-20f, 20f)] public float windZ = 0f;
        const float Spacing = 0.25f;
        const float H = 1f / 60f;

        Arena arena;
        floatBSR A;
        floatIC0 precond;
        bool built;
        float builtStiffness, builtMass;
        int builtW, builtH;

        NativeArray<float3> pos, vel;
        NativeArray<int2> edges;
        NativeArray<float> restLen;
        NativeArray<byte> pinned;
        NativeArray<float> outStats;   // [0] cg iters, [1] converged, [2] rnorm
        float frameMs;
        readonly Stopwatch sw = new Stopwatch();

        int NodeCount => gridWidth * gridHeight;

        void OnEnable() => Build();

        void OnDisable() => TearDown();

        void TearDown()
        {
            if (built) { arena.Dispose(); built = false; }
            if (pos.IsCreated) pos.Dispose();
            if (vel.IsCreated) vel.Dispose();
            if (edges.IsCreated) edges.Dispose();
            if (restLen.IsCreated) restLen.Dispose();
            if (pinned.IsCreated) pinned.Dispose();
            if (outStats.IsCreated) outStats.Dispose();
        }

        void Build()
        {
            TearDown();

            int W = gridWidth, Hn = gridHeight, n = NodeCount;
            pos = new NativeArray<float3>(n, Allocator.Persistent);
            vel = new NativeArray<float3>(n, Allocator.Persistent);
            pinned = new NativeArray<byte>(n, Allocator.Persistent);
            outStats = new NativeArray<float>(3, Allocator.Persistent);

            for (int j = 0; j < Hn; j++)
                for (int i = 0; i < W; i++)
                {
                    int id = j * W + i;
                    pos[id] = new float3((i - (W - 1) * 0.5f) * Spacing, 2f - j * Spacing, 0f);
                    pinned[id] = (byte)(j == 0 ? 1 : 0);
                }

            // structural + shear springs
            int edgeCount = (W - 1) * Hn + W * (Hn - 1) + 2 * (W - 1) * (Hn - 1);
            edges = new NativeArray<int2>(edgeCount, Allocator.Persistent);
            restLen = new NativeArray<float>(edgeCount, Allocator.Persistent);
            int e = 0;
            void AddEdge(int a, int b)
            {
                edges[e] = new int2(math.min(a, b), math.max(a, b));
                restLen[e] = math.distance(pos[a], pos[b]);
                e++;
            }
            for (int j = 0; j < Hn; j++)
                for (int i = 0; i < W; i++)
                {
                    int id = j * W + i;
                    if (i + 1 < W) AddEdge(id, id + 1);
                    if (j + 1 < Hn) AddEdge(id, id + W);
                    if (i + 1 < W && j + 1 < Hn) { AddEdge(id, id + W + 1); AddEdge(id + 1, id + W); }
                }

            // assemble A = M + h²·k·L once, symmetric LOWER-block storage
            arena = new Arena(Allocator.Persistent);
            float h2k = H * H * stiffness;
            var builder = new floatBSRBuilder(n, n, 3, 3, Allocator.Temp, edgeCount * 2 + n);
            var degree = new NativeArray<float>(n, Allocator.Temp);
            for (int k = 0; k < edgeCount; k++)
            {
                int a = edges[k].x, b = edges[k].y;   // a < b
                degree[a] += h2k; degree[b] += h2k;
                for (int d = 0; d < 3; d++)
                    builder.AddValue(3 * b + d, 3 * a + d, -h2k);   // lower off-diagonal block
            }
            for (int i = 0; i < n; i++)
            {
                float mi = pinned[i] == 1 ? 1e7f : nodeMass;
                for (int d = 0; d < 3; d++)
                    builder.AddValue(3 * i + d, 3 * i + d, mi + degree[i]);
            }
            degree.Dispose();

            A = builder.ToBSRSymmetric(ref arena);
            builder.Dispose();
            precond = arena.floatIC0(in A);

            built = true;
            builtStiffness = stiffness; builtMass = nodeMass;
            builtW = W; builtH = Hn;
        }

        void Update()
        {
            if (builtW != gridWidth || builtH != gridHeight
                || builtStiffness != stiffness || builtMass != nodeMass)
                Build();

            var job = new SpringStepJob
            {
                A = A, Precond = precond,
                Pos = pos, Vel = vel, Edges = edges, RestLen = restLen, Pinned = pinned,
                Out = outStats,
                Stiffness = stiffness, NodeMass = nodeMass, Damping = damping,
                WindZ = windZ, H = H,
            };

            sw.Restart();
            job.Run();
            sw.Stop();
            frameMs = (float)sw.Elapsed.TotalMilliseconds;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !edges.IsCreated) return;
            Gizmos.color = new Color(0.8f, 0.9f, 1f);
            for (int k = 0; k < edges.Length; k++)
                Gizmos.DrawLine((Vector3)pos[edges[k].x], (Vector3)pos[edges[k].y]);
            Gizmos.color = Color.red;
            for (int i = 0; i < pinned.Length; i++)
                if (pinned[i] == 1) Gizmos.DrawSphere((Vector3)pos[i], 0.03f);
        }

        void OnGUI()
        {
            int dof = NodeCount * 3;
            GUILayout.BeginArea(new Rect(10, 10, 400, 200), GUI.skin.box);
            GUILayout.Label($"Implicit springs — {NodeCount} nodes ({dof} dof), IC(0)-PCG, {frameMs:F2} ms/frame");
            GUILayout.Label($"cg iters: {outStats[0]:F0}   converged: {outStats[1] == 1f}   rnorm: {outStats[2]:E1}");
            stiffness = LabeledSlider($"stiffness {stiffness:F0}", stiffness, 10f, 800f);
            windZ = LabeledSlider($"wind {windZ:F1}", windZ, -20f, 20f);
            damping = LabeledSlider($"damping {damping:F2}", damping, 0f, 3f);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Poke center"))
            {
                int c = (gridHeight / 2) * gridWidth + gridWidth / 2;
                vel[c] += new float3(0f, 0f, 6f);
            }
            if (GUILayout.Button("Reset")) Build();
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        static float LabeledSlider(string label, float v, float lo, float hi)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(110));
            v = GUILayout.HorizontalSlider(v, lo, hi, GUILayout.Width(240));
            GUILayout.EndHorizontal();
            return v;
        }
    }

    /// <summary>
    /// One implicit step: nonlinear spring forces → rhs = M·v + h·f, then
    /// IC(0)-PCG solve of (M + h²kL) v+ = rhs, then integrate positions.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct SpringStepJob : IJob
    {
        [ReadOnly] public floatBSR A;
        [ReadOnly] public floatIC0 Precond;
        public NativeArray<float3> Pos, Vel;
        [ReadOnly] public NativeArray<int2> Edges;
        [ReadOnly] public NativeArray<float> RestLen;
        [ReadOnly] public NativeArray<byte> Pinned;
        public NativeArray<float> Out;
        public float Stiffness, NodeMass, Damping, WindZ, H;

        public void Execute()
        {
            int n = Pos.Length, dof = n * 3;
            var f = new NativeArray<float3>(n, Allocator.Temp);

            for (int i = 0; i < n; i++)
            {
                Vel[i] *= math.max(0f, 1f - Damping * H);
                f[i] = new float3(0f, -9.81f * NodeMass, WindZ * 0.01f * (1f + 0.3f * math.sin(Pos[i].x * 3f)));
            }

            for (int k = 0; k < Edges.Length; k++)
            {
                int a = Edges[k].x, b = Edges[k].y;
                float3 d = Pos[b] - Pos[a];
                float len = math.length(d);
                if (len < 1e-6f) continue;
                float3 fs = Stiffness * (len - RestLen[k]) * (d / len);
                f[a] += fs; f[b] -= fs;
            }

            var rhs = new floatN(dof, Allocator.Temp);
            var v = new floatN(dof, Allocator.Temp);
            for (int i = 0; i < n; i++)
            {
                float mi = Pinned[i] == 1 ? 1e7f : NodeMass;
                float3 fi = Pinned[i] == 1 ? float3.zero : f[i];
                float3 vi = Pinned[i] == 1 ? float3.zero : Vel[i];
                for (int d = 0; d < 3; d++)
                {
                    rhs[3 * i + d] = mi * vi[d] + H * fi[d];
                    v[3 * i + d] = vi[d];   // warm start from current velocity
                }
            }

            var r = new floatN(dof, Allocator.Temp);
            var p = new floatN(dof, Allocator.Temp);
            var Ap = new floatN(dof, Allocator.Temp);
            var z = new floatN(dof, Allocator.Temp);
            SolveInfo info = Krylov.cg(in A, in Precond, in rhs, ref v,
                                        ref r, ref p, ref Ap, ref z,
                                        200, 1e-5f);
            Out[0] = info.iterations;
            Out[1] = info ? 1f : 0f;
            Out[2] = (float)info.rnorm;

            for (int i = 0; i < n; i++)
            {
                if (Pinned[i] == 1) { Vel[i] = float3.zero; continue; }
                var vi = new float3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
                Vel[i] = vi;
                Pos[i] += H * vi;
            }
        }
    }
}
