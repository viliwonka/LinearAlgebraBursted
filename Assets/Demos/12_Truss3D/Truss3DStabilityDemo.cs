using System.Collections.Generic;
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
    /// Structural stability of a parametric 3D lattice tower (square-section space frame,
    /// <see cref="stories"/> stories tall). The stiffness matrix is assembled into 3×3-block
    /// symmetric BSR (lower-block storage, 3 dof/node); a Burst job runs IC(0)-preconditioned
    /// LOBPCG (<see cref="TrussEigenJobIC0"/>) for the 4 smallest eigenpairs every frame
    /// (warm-started from the previous frame's cache). IC(0) captures the inter-story coupling
    /// of the slender tower — block-Jacobi (the 2D house frame's preconditioner) sees only each
    /// node's own diagonal block and cannot resolve the global sway/torsion mode that is the
    /// softest eigenvector here. Toggle a story's diagonal face-bracing off
    /// and watch lambda1 collapse toward a shear/torsion mechanism at that story; the
    /// corresponding mode shape is animated on the frame.
    /// </summary>
    public class Truss3DStabilityDemo : MonoBehaviour
    {
        [Range(1, 24)] public int stories = 8;
        [Range(0.5f, 20f)] public float stiffnessEA = 8f;
        [Range(0f, 0.5f)] public float modeAmplitude = 0.15f;
        [Range(0, 3)] public int shownMode;

        const float FootprintWidth = 1f;
        const float StoryHeight = 1f;
        const int K = 4;

        // per-story diagonal bracing (4 face-diagonals per story, toggled together --
        // dropping a whole story's bracing is what turns it into a mechanism).
        public bool[] braceOn;

        float3[] Nodes;
        int2[] Chords;      // vertical, 4 per story
        int2[] Rings;       // horizontal, 4 per level
        int2[] Diagonals;   // one per face per story, grouped 4 per story (index = story*4 + face)

        int N => Nodes.Length * 3;

        Arena arena;
        floatBSR A;
        floatIC0 precond;
        floatLOBPCGCache cache;
        floatN lambda;      // arena-owned view of cache.lambda after solve
        floatMxN modes;     // arena-owned view of cache.X (K x N)
        bool built;
        float builtEA;
        int builtStories;
        bool[] builtBraces;
        NativeArray<float> outStats;   // [0] iterations, [1] converged
        float frameMs;
        readonly Stopwatch sw = new Stopwatch();
        GUIStyle stabilityLabelStyle;   // lazily built once; only its textColor is mutated per frame

        void OnEnable()
        {
            outStats = new NativeArray<float>(2, Allocator.Persistent);
            braceOn = NewBraceArray(stories, null);
            builtBraces = (bool[])braceOn.Clone();
            Build();
        }

        void OnDisable()
        {
            if (built) { arena.Dispose(); built = false; }
            if (outStats.IsCreated) outStats.Dispose();
        }

        static bool[] NewBraceArray(int n, bool[] old)
        {
            var a = new bool[n];
            for (int i = 0; i < n; i++)
                a[i] = (old != null && i < old.Length) ? old[i] : true;
            return a;
        }

        // Square-section lattice tower: 4 corner nodes per level, stories+1 levels. Members:
        // vertical chords, horizontal ring beams every level, and one diagonal per face per
        // story (all leaning the same rotational sense -- triangulates every face against
        // both racking shear and torsion).
        void BuildGeometry()
        {
            int levels = stories + 1;
            Nodes = new float3[levels * 4];

            float hw = FootprintWidth * 0.5f;
            float2[] corner =
            {
                new float2(-hw, -hw), new float2(hw, -hw), new float2(hw, hw), new float2(-hw, hw),
            };
            for (int l = 0; l < levels; l++)
                for (int c = 0; c < 4; c++)
                    Nodes[l * 4 + c] = new float3(corner[c].x, l * StoryHeight, corner[c].y);

            var chords = new List<int2>(stories * 4);
            for (int l = 0; l < stories; l++)
                for (int c = 0; c < 4; c++)
                    chords.Add(new int2(l * 4 + c, (l + 1) * 4 + c));
            Chords = chords.ToArray();

            var rings = new List<int2>(levels * 4);
            for (int l = 0; l < levels; l++)
                for (int c = 0; c < 4; c++)
                    rings.Add(new int2(l * 4 + c, l * 4 + (c + 1) % 4));
            Rings = rings.ToArray();

            var diagonals = new List<int2>(stories * 4);
            for (int l = 0; l < stories; l++)
                for (int f = 0; f < 4; f++)
                    diagonals.Add(new int2(l * 4 + f, (l + 1) * 4 + (f + 1) % 4));
            Diagonals = diagonals.ToArray();
        }

        void Build()
        {
            if (built) arena.Dispose();
            arena = new Arena(Allocator.Persistent);

            BuildGeometry();

            int nb = Nodes.Length;
            int capHint = (Chords.Length + Rings.Length + Diagonals.Length + 4) * 27;
            var builder = new floatBSRBuilder(nb, nb, 3, 3, Allocator.Temp, capHint);

            void AddBar(int a, int b)
            {
                float3 d = Nodes[b] - Nodes[a];
                float L = math.length(d);
                float3 u = d / L;
                float k = stiffnessEA / L;
                // 3×3 block k·uuT on both diagonals, -k·uuT at the LOWER (max,min) position --
                // same pattern as TrussStabilityDemo.Build, one more row/col for the 3rd dof.
                int lo = math.min(a, b), hi = math.max(a, b);
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                    {
                        float v = k * u[r] * u[c];
                        builder.AddValue(3 * a + r, 3 * a + c, v);
                        builder.AddValue(3 * b + r, 3 * b + c, v);
                        builder.AddValue(3 * hi + r, 3 * lo + c, -v);
                    }
            }

            foreach (var m in Chords) AddBar(m.x, m.y);
            foreach (var m in Rings) AddBar(m.x, m.y);
            for (int s = 0; s < stories; s++)
                if (braceOn[s])
                    for (int f = 0; f < 4; f++)
                        AddBar(Diagonals[s * 4 + f].x, Diagonals[s * 4 + f].y);

            // pinned supports at the 4 base-level nodes: penalty on their diagonal blocks.
            // Keep the penalty within ~3 decades of the bar stiffness (~1e3 for float), NOT
            // 1e6 (see TrussStabilityDemo.Build) -- float LOBPCG forms Gram matrices ~penalty²
            // and a huge penalty pushes the real O(EA) eigenvalues below the float noise floor.
            for (int c = 0; c < 4; c++)
                for (int d = 0; d < 3; d++)
                    builder.AddValue(3 * c + d, 3 * c + d, 1e3f);

            A = builder.ToBSRSymmetric(ref arena);
            builder.Dispose();
            precond = arena.floatIC0(in A);
            cache = arena.floatLOBPCGCache(N, K);

            built = true;
            builtEA = stiffnessEA;
            builtStories = stories;
            builtBraces = (bool[])braceOn.Clone();
        }

        void Update()
        {
            if (stories != builtStories)
                braceOn = NewBraceArray(stories, braceOn);

            bool dirty = stories != builtStories || builtEA != stiffnessEA;
            if (!dirty)
                for (int i = 0; i < braceOn.Length; i++) dirty |= braceOn[i] != builtBraces[i];
            if (dirty) Build();

            var job = new TrussEigenJobIC0
            {
                Op = new floatBSROperator(in A),
                Precond = precond,
                Cache = cache,
                Out = outStats,
                K = K,
            };

            sw.Restart();
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
                float3 p = Nodes[node];
                float dx = modes[shownMode, 3 * node] * wob;
                float dy = modes[shownMode, 3 * node + 1] * wob;
                float dz = modes[shownMode, 3 * node + 2] * wob;
                return new Vector3(p.x + dx, p.y + dy, p.z + dz);
            }

            Gizmos.color = Color.white;
            foreach (var m in Chords) Gizmos.DrawLine(P(m.x), P(m.y));
            foreach (var m in Rings) Gizmos.DrawLine(P(m.x), P(m.y));

            Gizmos.color = Color.yellow;
            for (int s = 0; s < stories; s++)
                if (braceOn[s])
                    for (int f = 0; f < 4; f++)
                    {
                        var m = Diagonals[s * 4 + f];
                        Gizmos.DrawLine(P(m.x), P(m.y));
                    }

            Gizmos.color = Color.red;
            for (int c = 0; c < 4; c++) Gizmos.DrawSphere(P(c), 0.06f);
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 460, 340), GUI.skin.box);
            GUILayout.Label($"3D truss tower — LOBPCG k={K} over {N}-dof BSR (3x3 blocks), {frameMs:F2} ms/frame");
            GUILayout.Label($"iters: {outStats[0]:F0} (warm)   converged: {outStats[1] == 1f}");
            if (built && lambda.IsCreated)
            {
                GUILayout.Label($"lambda = [{lambda[0]:F3}, {lambda[1]:F3}, {lambda[2]:F3}, {lambda[3]:F3}]");
                bool unstable = lambda[0] < 0.05f * stiffnessEA;
                if (stabilityLabelStyle == null) stabilityLabelStyle = new GUIStyle(GUI.skin.label);
                stabilityLabelStyle.normal.textColor = unstable ? Color.red : Color.green;
                GUILayout.Label(unstable
                    ? "lambda1 ≈ 0 — near-mechanism, tower is UNSTABLE"
                    : "tower is stiff (no soft modes)", stabilityLabelStyle);
            }

            GUILayout.Label("Diagonal bracing (per story):");
            for (int row = 0; row < braceOn.Length; row += 8)
            {
                GUILayout.BeginHorizontal();
                for (int i = row; i < math.min(row + 8, braceOn.Length); i++)
                    braceOn[i] = GUILayout.Toggle(braceOn[i], $"s{i}");
                GUILayout.EndHorizontal();
            }

            shownMode = (int)LabeledSlider($"mode {shownMode} (lambda={((built && lambda.IsCreated) ? lambda[shownMode] : 0f):F3})", shownMode, 0, 3.49f);
            stiffnessEA = LabeledSlider($"EA {stiffnessEA:F1}", stiffnessEA, 0.5f, 20f);
            modeAmplitude = LabeledSlider($"amplitude {modeAmplitude:F2}", modeAmplitude, 0f, 0.5f);
            stories = (int)LabeledSlider($"stories {stories}", stories, 1, 24.49f);
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

    /// <summary>Warm LOBPCG smallest-k eigenpairs of the tower stiffness matrix with an IC(0)
    /// preconditioner. IC(0) factors A's lower block pattern, so it carries the inter-story
    /// coupling that resolves the tower's global sway/torsion mode — the softest eigenvector,
    /// which the diagonal-only block-Jacobi of the 2D house frame cannot see.</summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct TrussEigenJobIC0 : IJob
    {
        [ReadOnly] public floatBSROperator Op;
        [ReadOnly] public floatIC0 Precond;
        public floatLOBPCGCache Cache;
        public NativeArray<float> Out;
        public int K;

        public void Execute()
        {
            LOBPCGInfo info = Eigen.lobpcg(in Op, in Precond, ref Cache, K);
            Out[0] = info.iterations;
            Out[1] = info ? 1f : 0f;
        }
    }
}
