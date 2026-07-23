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
        public bool colorByStress = true;   // color members by modal axial force (where it breaks first)
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

        floatBSR A;
        floatBlockJacobi mJacobi;   // only the field matching builtPrecond is live each Build()
        floatIC0 mIC0;
        floatSSOR mSSOR;
        floatLOBPCGCache cache;
        floatN lambda;      // view of cache.lambda after solve
        floatMxN modes;     // view of cache.X (K x N)
        floatN residX, residAx;   // scratch for per-mode residuals
        float softestResidual;    // ||A x0 - lambda0 x0|| / ||A x0|| of the current softest mode
        float shownModeResidual;  // same, for the currently displayed mode
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
            if (built) { A.Dispose(); DisposePrecond(); cache.Dispose(); residX.Dispose(); residAx.Dispose(); built = false; }
            if (outStats.IsCreated) outStats.Dispose();
        }

        // Only the field matching builtPrecond (the preconditioner live from the last Build()) was
        // ever constructed -- dispose that one field only.
        void DisposePrecond()
        {
            switch (builtPrecond)
            {
                case Preconditioner.BlockJacobi: mJacobi.Dispose(); break;
                case Preconditioner.SSOR:        mSSOR.Dispose(); break;
                default:                         mIC0.Dispose(); break;
            }
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
            if (built) { A.Dispose(); DisposePrecond(); cache.Dispose(); residX.Dispose(); residAx.Dispose(); }

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

            A = builder.ToBSRSymmetric(Allocator.Persistent);
            builder.Dispose();

            switch (preconditioner)
            {
                case Preconditioner.BlockJacobi: mJacobi = new floatBlockJacobi(in A, Allocator.Persistent); break;
                case Preconditioner.SSOR:        mSSOR = new floatSSOR(in A, Allocator.Persistent); break;
                default:                         mIC0 = new floatIC0(in A, Allocator.Persistent); break;
            }
            cache = new floatLOBPCGCache(N, K + Guard, Allocator.Persistent);
            residX = new floatN(N, Allocator.Persistent);
            residAx = new floatN(N, Allocator.Persistent);

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

            // Per-mode residuals, recomputed independently via spMV (not the solver's self-report):
            // small for a genuine eigenpair, ~1 for a mechanism the solver cannot pin (a singular
            // stiffness has exactly-null modes whose eigenvectors are undefined). softest drives the
            // stable/unstable verdict; shown gates the mode animation.
            softestResidual = ModeResidual(0);
            shownModeResidual = ModeResidual(shownMode);
        }

        // ||A x_m - lambda_m x_m|| / ||A x_m|| for cache mode m, via an independent spMV.
        float ModeResidual(int m)
        {
            for (int i = 0; i < N; i++) residX[i] = cache.X[m, i];
            BSR.spMV(in A, in residX, ref residAx);
            float num = 0f, den = 0f;
            for (int i = 0; i < N; i++)
            {
                float r = residAx[i] - cache.lambda[m] * residX[i];
                num += r * r;
                den += residAx[i] * residAx[i];
            }
            return math.sqrt(num) / (math.sqrt(den) + 1e-30f);
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !built || !lambda.IsCreated) return;

            // A mechanism (unconverged null) mode has no defined shape and would just jitter -- freeze
            // it undeformed and fall back to structural colors rather than animate noise.
            bool resolved = shownModeResidual <= 0.25f;
            bool stress = colorByStress && resolved;
            float wob = resolved ? modeAmplitude * math.sin(6f * Time.time) : 0f;
            Vector3 P(int node)
            {
                float3 p = Nodes[node];
                float dx = modes[shownMode, 3 * node] * wob;
                float dy = modes[shownMode, 3 * node + 1] * wob;
                float dz = modes[shownMode, 3 * node + 2] * wob;
                return new Vector3(p.x + dx, p.y + dy, p.z + dz);
            }

            // Modal axial force of a member in the shown mode: (EA/L) * elongation, elongation =
            // (disp_b - disp_a) . axis_unit. The member with the largest |force| is where the mode
            // concentrates stress -- where the structure yields/buckles first. The mode shape sets
            // only the RELATIVE distribution (eigenvectors are unit-normalized), so normalize to the
            // per-mode peak.
            float Force(int a, int b)
            {
                float3 axis = Nodes[b] - Nodes[a];
                float L = math.length(axis);
                float3 u = axis / L;
                float3 da = new float3(modes[shownMode, 3 * a], modes[shownMode, 3 * a + 1], modes[shownMode, 3 * a + 2]);
                float3 db = new float3(modes[shownMode, 3 * b], modes[shownMode, 3 * b + 1], modes[shownMode, 3 * b + 2]);
                return (stiffnessEA / L) * math.dot(db - da, u);
            }

            float maxF = 1e-30f;
            if (stress)
            {
                foreach (var m in Columns) maxF = math.max(maxF, math.abs(Force(m.x, m.y)));
                foreach (var m in Beams) maxF = math.max(maxF, math.abs(Force(m.x, m.y)));
                foreach (var m in Diaphragm) maxF = math.max(maxF, math.abs(Force(m.x, m.y)));
                for (int k = 0; k < Braces.Length; k++)
                    if (braceOn[BraceStory[k]]) maxF = math.max(maxF, math.abs(Force(Braces[k].x, Braces[k].y)));
            }

            // blue (low) -> cyan -> green -> yellow -> red (peak) via hue 0.66..0.
            void Draw(int a, int b, Color baseColor)
            {
                Gizmos.color = stress
                    ? Color.HSVToRGB((1f - math.saturate(math.abs(Force(a, b)) / maxF)) * 0.66f, 0.9f, 1f)
                    : baseColor;
                Gizmos.DrawLine(P(a), P(b));
            }

            foreach (var m in Columns) Draw(m.x, m.y, Color.white);
            foreach (var m in Beams) Draw(m.x, m.y, new Color(0.6f, 0.6f, 0.6f));
            foreach (var m in Diaphragm) Draw(m.x, m.y, new Color(0.25f, 0.5f, 0.55f));
            for (int k = 0; k < Braces.Length; k++)
                if (braceOn[BraceStory[k]]) Draw(Braces[k].x, Braces[k].y, Color.yellow);

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
                GUILayout.Label($"lambda = [{Lam(lambda[0])}, {Lam(lambda[1])}, {Lam(lambda[2])}, {Lam(lambda[3])}]");
                // A braced building's genuine fundamental (lateral sway) mode is legitimately SMALL
                // relative to EA, so an absolute lambda bar flags every real frame as unstable. Two
                // signs mark a mechanism instead: (1) lambda0 at the float noise floor (a singular
                // stiffness's exactly-null mode -- caught even when the solver returns a CLEAN null
                // vector with ~0 residual), or (2) the softest mode's own residual ||A x0 - lambda0
                // x0|| / ||A x0|| near 1 (a near-null mode the solver cannot pin). Together these are
                // robust to building size and to HOW MANY null modes exist (a spectral-gap test
                // misreads >=K null modes as "stable"; a residual-only test misses clean null modes).
                bool nullMode = math.abs(lambda[0]) < 1e-5f * stiffnessEA;
                bool unstable = nullMode || !(softestResidual <= 0.25f);   // NaN-safe
                if (stabilityLabelStyle == null) stabilityLabelStyle = new GUIStyle(GUI.skin.label);
                stabilityLabelStyle.normal.textColor = unstable ? Color.red : Color.green;
                GUILayout.Label(unstable
                    ? $"mechanism — softest mode lambda0={Lam(lambda[0])}, residual {softestResidual:F2} — UNSTABLE"
                    : $"stable — softest sway lambda0={lambda[0]:E2}, residual {softestResidual:F3}", stabilityLabelStyle);
            }

            colorByStress = GUILayout.Toggle(colorByStress, "colour members by modal stress (red = breaks first)");

            GUILayout.Label("Perimeter bracing (per story):");
            for (int row = 0; row < braceOn.Length; row += 10)
            {
                GUILayout.BeginHorizontal();
                for (int i = row; i < math.min(row + 10, braceOn.Length); i++)
                    braceOn[i] = GUILayout.Toggle(braceOn[i], $"{i}");
                GUILayout.EndHorizontal();
            }

            string modeTag = (built && lambda.IsCreated)
                ? $"lambda={Lam(lambda[shownMode])}{(shownModeResidual > 0.25f ? ", mechanism — shape undefined" : "")}"
                : "lambda=0";
            shownMode = (int)LabeledSlider($"mode {shownMode} ({modeTag})", shownMode, 0, 3.49f);
            stiffnessEA = LabeledSlider($"EA {stiffnessEA:F1}", stiffnessEA, 0.5f, 20f);
            modeAmplitude = LabeledSlider($"amplitude {modeAmplitude:F2}", modeAmplitude, 0f, 0.5f);
            baysX = (int)LabeledSlider($"bays X {baysX}", baysX, 1, 10.49f);
            baysZ = (int)LabeledSlider($"bays Z {baysZ}", baysZ, 1, 10.49f);
            stories = (int)LabeledSlider($"stories {stories}", stories, 1, 60.49f);
            GUILayout.EndArea();
        }

        // Render an eigenvalue: values at the float noise floor (a singular stiffness's exactly-null
        // mechanism modes) read as "~0" instead of flickering their ~1e-7 noise in the display.
        string Lam(float v) => math.abs(v) < 1e-5f * stiffnessEA ? "~0" : v.ToString("E2");

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
