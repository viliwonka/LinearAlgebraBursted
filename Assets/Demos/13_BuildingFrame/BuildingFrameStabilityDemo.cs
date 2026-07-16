using System.Collections.Generic;
using System.Diagnostics;
using LinearAlgebra;
using LinearAlgebra.Sparse;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// Structural stability of a parametric braced-frame building: a <see cref="baysX"/>×
    /// <see cref="baysZ"/>-bay, <see cref="stories"/>-story space frame (columns on every grid
    /// line, two-way floor beams at every level, and per-story diagonal bracing on the four
    /// perimeter walls). The stiffness matrix is assembled into 3×3-block symmetric BSR (3 dof/
    /// node) and a Burst job runs preconditioned LOBPCG for the 4 smallest eigenpairs every frame,
    /// warm-started from the previous frame's cache. This is the tower demo scaled to a real
    /// building: at 8×8 bays × 40 stories the system is ~10k dof, the regime where the
    /// preconditioner choice actually bites — IC(0)'s forward/backward triangular solve is
    /// serial, block-Jacobi's diagonal apply is fully parallel, and the cold-iteration readout
    /// lets you watch that trade play out as you switch. Drop a story's perimeter bracing and its
    /// softest eigenvalue collapses toward a soft-story sway/torsion mechanism; the mode shape is
    /// animated on the frame.
    /// </summary>
    public class BuildingFrameStabilityDemo : MonoBehaviour
    {
        public enum Preconditioner { BlockJacobi, IC0, SSOR }

        [Range(1, 10)] public int baysX = 4;
        [Range(1, 10)] public int baysZ = 4;
        [Range(1, 60)] public int stories = 16;
        [Range(0.5f, 20f)] public float stiffnessEA = 8f;
        [Range(0f, 0.5f)] public float modeAmplitude = 0.15f;
        [Range(0, 3)] public int shownMode;
        public Preconditioner preconditioner = Preconditioner.IC0;

        const float BayWidth = 1f;
        const float BayDepth = 1f;
        const float StoryHeight = 1f;
        const int K = 4;
        // guard ("ghost") vectors: LOBPCG iterates on K+Guard vectors but returns the K smallest.
        // A doubly-symmetric building has a near-degenerate soft cluster (X-sway ≈ Z-sway ≈
        // torsion); the extra guard room gives the wanted pairs spectral separation from the rest
        // of the cluster, the standard LOBPCG aid for converging a degenerate bottom faster.
        const int Guard = 4;

        // per-story perimeter bracing (all four walls of a story toggled together -- dropping a
        // story's bracing is what turns that level into a soft-story mechanism).
        public bool[] braceOn;

        float3[] Nodes;
        int2[] Columns;     // vertical, one per grid line per story
        int2[] Beams;       // two-way floor beams, every level >= 1
        int2[] Diaphragm;   // one in-plane diagonal per floor panel -- rigid-diaphragm bracing
                            // (always on; without it a pin-jointed floor grid shears as a mechanism)
        int2[] Braces;      // perimeter wall diagonals; BraceStory[k] gives the owning story
        int[] BraceStory;

        int NW => baysX + 1;   // node columns across width
        int ND => baysZ + 1;   // node columns across depth
        int NodeIdx(int i, int j, int l) => (l * ND + j) * NW + i;
        int N => Nodes.Length * 3;

        Arena arena;
        floatBSR A;
        floatBlockJacobi mJacobi;   // only the field matching builtPrecond is live each Build()
        floatIC0 mIC0;
        floatSSOR mSSOR;
        floatLOBPCGCache cache;
        floatN lambda;      // arena-owned view of cache.lambda after solve
        floatMxN modes;     // arena-owned view of cache.X (K x N)
        bool built;
        float builtEA;
        int builtBaysX, builtBaysZ, builtStories;
        bool[] builtBraces;
        Preconditioner builtPrecond;
        NativeArray<float> outStats;   // [0] iterations, [1] converged
        float coldIters;    // iteration count of the first (cold) solve after a rebuild
        bool justBuilt;     // set by Build(), consumed by the next Update() to latch coldIters
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

        // Grid of NW x ND corner nodes per level, stories+1 levels. Members: vertical columns on
        // every grid line, two-way floor beams (X and Z) at every level >= 1, and one diagonal per
        // perimeter wall panel per story (all leaning the same rotational sense -- braces the
        // exterior against racking shear and torsion, like a braced tube).
        void BuildGeometry()
        {
            int levels = stories + 1;
            Nodes = new float3[NW * ND * levels];
            for (int l = 0; l < levels; l++)
                for (int j = 0; j < ND; j++)
                    for (int i = 0; i < NW; i++)
                        Nodes[NodeIdx(i, j, l)] = new float3(i * BayWidth, l * StoryHeight, j * BayDepth);

            var cols = new List<int2>(NW * ND * stories);
            for (int l = 0; l < stories; l++)
                for (int j = 0; j < ND; j++)
                    for (int i = 0; i < NW; i++)
                        cols.Add(new int2(NodeIdx(i, j, l), NodeIdx(i, j, l + 1)));
            Columns = cols.ToArray();

            var beams = new List<int2>();
            for (int l = 1; l < levels; l++)
            {
                for (int j = 0; j < ND; j++)
                    for (int i = 0; i < baysX; i++)
                        beams.Add(new int2(NodeIdx(i, j, l), NodeIdx(i + 1, j, l)));      // X beams
                for (int j = 0; j < baysZ; j++)
                    for (int i = 0; i < NW; i++)
                        beams.Add(new int2(NodeIdx(i, j, l), NodeIdx(i, j + 1, l)));      // Z beams
            }
            Beams = beams.ToArray();

            // rigid-diaphragm bracing: one in-plane diagonal per floor panel at every level >= 1,
            // triangulating the horizontal grid so it cannot shear as a parallelogram mechanism.
            var diaphragm = new List<int2>();
            for (int l = 1; l < levels; l++)
                for (int j = 0; j < baysZ; j++)
                    for (int i = 0; i < baysX; i++)
                        diaphragm.Add(new int2(NodeIdx(i, j, l), NodeIdx(i + 1, j + 1, l)));
            Diaphragm = diaphragm.ToArray();

            var braces = new List<int2>();
            var braceStory = new List<int>();
            for (int s = 0; s < stories; s++)
            {
                // front wall j=0 and back wall j=baysZ: diagonals span a bay in X, rising one story
                for (int i = 0; i < baysX; i++)
                {
                    braces.Add(new int2(NodeIdx(i, 0, s), NodeIdx(i + 1, 0, s + 1)));       braceStory.Add(s);
                    braces.Add(new int2(NodeIdx(i, baysZ, s), NodeIdx(i + 1, baysZ, s + 1))); braceStory.Add(s);
                }
                // left wall i=0 and right wall i=baysX: diagonals span a bay in Z, rising one story
                for (int j = 0; j < baysZ; j++)
                {
                    braces.Add(new int2(NodeIdx(0, j, s), NodeIdx(0, j + 1, s + 1)));       braceStory.Add(s);
                    braces.Add(new int2(NodeIdx(baysX, j, s), NodeIdx(baysX, j + 1, s + 1))); braceStory.Add(s);
                }
            }
            Braces = braces.ToArray();
            BraceStory = braceStory.ToArray();
        }

        void Build()
        {
            if (built) arena.Dispose();
            arena = new Arena(Allocator.Persistent);

            BuildGeometry();

            int nb = Nodes.Length;
            int capHint = (Columns.Length + Beams.Length + Diaphragm.Length + Braces.Length + ND * NW) * 27;
            var builder = new floatBSRBuilder(nb, nb, 3, 3, Allocator.Temp, capHint);

            void AddBar(int a, int b)
            {
                float3 d = Nodes[b] - Nodes[a];
                float L = math.length(d);
                float3 u = d / L;
                float k = stiffnessEA / L;
                // 3×3 block k·uuT on both diagonals, -k·uuT at the LOWER (max,min) position.
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

            foreach (var m in Columns) AddBar(m.x, m.y);
            foreach (var m in Beams) AddBar(m.x, m.y);
            foreach (var m in Diaphragm) AddBar(m.x, m.y);
            for (int k = 0; k < Braces.Length; k++)
                if (braceOn[BraceStory[k]]) AddBar(Braces[k].x, Braces[k].y);

            // pinned supports at every ground-level node: penalty on their diagonal blocks. Keep
            // the penalty within ~3 decades of the bar stiffness (~1e3 for float), NOT 1e6 -- float
            // LOBPCG forms Gram matrices ~penalty² and a huge penalty pushes the real O(EA)
            // eigenvalues below the float noise floor.
            for (int j = 0; j < ND; j++)
                for (int i = 0; i < NW; i++)
                {
                    int node = NodeIdx(i, j, 0);
                    for (int d = 0; d < 3; d++)
                        builder.AddValue(3 * node + d, 3 * node + d, 1e3f);
                }

            A = builder.ToBSRSymmetric(ref arena);
            builder.Dispose();

            switch (preconditioner)
            {
                case Preconditioner.BlockJacobi: mJacobi = arena.floatBlockJacobi(in A); break;
                case Preconditioner.SSOR:        mSSOR = new floatSSOR(in A, ref arena); break;
                default:                         mIC0 = arena.floatIC0(in A); break;
            }
            cache = arena.floatLOBPCGCache(N, K + Guard);

            built = true;
            justBuilt = true;
            builtEA = stiffnessEA;
            builtBaysX = baysX;
            builtBaysZ = baysZ;
            builtStories = stories;
            builtBraces = (bool[])braceOn.Clone();
            builtPrecond = preconditioner;
        }

        void Update()
        {
            if (stories != builtStories)
                braceOn = NewBraceArray(stories, braceOn);

            bool dirty = stories != builtStories || baysX != builtBaysX || baysZ != builtBaysZ
                         || builtEA != stiffnessEA || preconditioner != builtPrecond;
            if (!dirty)
                for (int i = 0; i < braceOn.Length; i++) dirty |= braceOn[i] != builtBraces[i];
            if (dirty) Build();

            var Op = new floatBSROperator(in A);
            sw.Restart();
            switch (builtPrecond)
            {
                case Preconditioner.BlockJacobi:
                {
                    var job = new TrussEigenJob { Op = Op, Precond = mJacobi, Cache = cache, Out = outStats, K = K };
                    IJobExtensions.RunByRef(ref job);
                    cache = job.Cache;
                    break;
                }
                case Preconditioner.SSOR:
                {
                    var job = new TrussEigenJobSSOR { Op = Op, Precond = mSSOR, Cache = cache, Out = outStats, K = K };
                    IJobExtensions.RunByRef(ref job);
                    cache = job.Cache;
                    break;
                }
                default:
                {
                    var job = new TrussEigenJobIC0 { Op = Op, Precond = mIC0, Cache = cache, Out = outStats, K = K };
                    IJobExtensions.RunByRef(ref job);
                    cache = job.Cache;
                    break;
                }
            }
            sw.Stop();
            frameMs = (float)sw.Elapsed.TotalMilliseconds;

            // The per-frame solve is warm (starts from the previous frame's eigenvectors), so its
            // iteration count collapses at steady state. The cold count -- the solve on the frame
            // right after a rebuild -- is where the preconditioners visibly differ, so latch it.
            if (justBuilt) { coldIters = outStats[0]; justBuilt = false; }

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
            foreach (var m in Columns) Gizmos.DrawLine(P(m.x), P(m.y));
            Gizmos.color = new Color(0.6f, 0.6f, 0.6f);
            foreach (var m in Beams) Gizmos.DrawLine(P(m.x), P(m.y));
            Gizmos.color = new Color(0.25f, 0.5f, 0.55f);
            foreach (var m in Diaphragm) Gizmos.DrawLine(P(m.x), P(m.y));

            Gizmos.color = Color.yellow;
            for (int k = 0; k < Braces.Length; k++)
                if (braceOn[BraceStory[k]]) Gizmos.DrawLine(P(Braces[k].x), P(Braces[k].y));

            Gizmos.color = Color.red;
            for (int j = 0; j < ND; j++)
                for (int i = 0; i < NW; i++)
                    Gizmos.DrawSphere(P(NodeIdx(i, j, 0)), 0.05f);
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 470, 420), GUI.skin.box);
            GUILayout.Label($"Building frame {baysX}×{baysZ} bays × {stories} stories — LOBPCG k={K} over {N}-dof BSR, {frameMs:F2} ms/frame");
            GUILayout.Label($"iters: {coldIters:F0} cold / {outStats[0]:F0} warm   converged: {outStats[1] == 1f}");

            GUILayout.BeginHorizontal();
            GUILayout.Label("precond:", GUILayout.Width(60));
            foreach (Preconditioner p in System.Enum.GetValues(typeof(Preconditioner)))
                if (GUILayout.Toggle(preconditioner == p, p.ToString(), GUI.skin.button) && preconditioner != p)
                    preconditioner = p;
            GUILayout.EndHorizontal();

            if (built && lambda.IsCreated)
            {
                GUILayout.Label($"lambda = [{lambda[0]:F3}, {lambda[1]:F3}, {lambda[2]:F3}, {lambda[3]:F3}]");
                bool unstable = lambda[0] < 0.05f * stiffnessEA;
                if (stabilityLabelStyle == null) stabilityLabelStyle = new GUIStyle(GUI.skin.label);
                stabilityLabelStyle.normal.textColor = unstable ? Color.red : Color.green;
                GUILayout.Label(unstable
                    ? "lambda1 ≈ 0 — soft-story mechanism, building is UNSTABLE"
                    : "building is stiff (no soft modes)", stabilityLabelStyle);
            }

            GUILayout.Label("Perimeter bracing (per story):");
            for (int row = 0; row < braceOn.Length; row += 10)
            {
                GUILayout.BeginHorizontal();
                for (int i = row; i < math.min(row + 10, braceOn.Length); i++)
                    braceOn[i] = GUILayout.Toggle(braceOn[i], $"{i}");
                GUILayout.EndHorizontal();
            }

            shownMode = (int)LabeledSlider($"mode {shownMode} (lambda={((built && lambda.IsCreated) ? lambda[shownMode] : 0f):F3})", shownMode, 0, 3.49f);
            stiffnessEA = LabeledSlider($"EA {stiffnessEA:F1}", stiffnessEA, 0.5f, 20f);
            modeAmplitude = LabeledSlider($"amplitude {modeAmplitude:F2}", modeAmplitude, 0f, 0.5f);
            baysX = (int)LabeledSlider($"bays X {baysX}", baysX, 1, 10.49f);
            baysZ = (int)LabeledSlider($"bays Z {baysZ}", baysZ, 1, 10.49f);
            stories = (int)LabeledSlider($"stories {stories}", stories, 1, 60.49f);
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
}
