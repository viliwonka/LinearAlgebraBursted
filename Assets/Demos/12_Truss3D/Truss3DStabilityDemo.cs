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
    /// symmetric BSR (lower-block storage, 3 dof/node); a Burst job runs preconditioned LOBPCG
    /// for the 4 smallest eigenpairs every frame (warm-started from the previous frame's cache).
    /// The preconditioner is switchable at runtime (block-Jacobi / IC(0) / SSOR) and the cold
    /// iteration count is displayed so their strength is directly comparable: on the slender
    /// tower IC(0) and SSOR capture the inter-story coupling that resolves the global
    /// sway/torsion mode — the softest eigenvector — while block-Jacobi sees only each node's own
    /// diagonal block and needs several times as many iterations to reach it. Toggle a story's
    /// diagonal face-bracing off and watch lambda1 collapse toward a shear/torsion mechanism at
    /// that story; the corresponding mode shape is animated on the frame.
    /// </summary>
    public class Truss3DStabilityDemo : MonoBehaviour
    {
        public enum Preconditioner { BlockJacobi, IC0, SSOR }

        [Range(1, 24)] public int stories = 8;
        [Range(0.5f, 20f)] public float stiffnessEA = 8f;
        [Range(0f, 0.5f)] public float modeAmplitude = 0.15f;
        [Range(0, 3)] public int shownMode;
        public bool colorByStress = true;   // color members by modal axial force (where it breaks first)
        public Preconditioner preconditioner = Preconditioner.IC0;

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
        int2[] Diaphragms;  // one in-plane diagonal per floor (level >= 1), always present

        int N => Nodes.Length * 3;

        Arena arena;
        floatBSR A;
        floatBlockJacobi mJacobi;   // only the field matching builtPrecond is live each Build()
        floatIC0 mIC0;
        floatSSOR mSSOR;
        floatLOBPCGCache cache;
        floatN lambda;      // arena-owned view of cache.lambda after solve
        floatMxN modes;     // arena-owned view of cache.X (K x N)
        floatN residX, residAx;   // scratch for per-mode residuals (arena-owned)
        float softestResidual;    // ||A x0 - lambda0 x0|| / ||A x0|| of the current softest mode
        float shownModeResidual;  // same, for the currently displayed mode
        bool built;
        float builtEA;
        int builtStories;
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

            // Floor-plane (diaphragm) bracing: one in-plane diagonal per suspended floor. Without it
            // each square floor ring is a rhombus mechanism in its own plane, and the box lozenges in
            // plan regardless of how the vertical faces are braced -- a spurious near-zero mode that
            // masks the real stable/mechanism transition the face braces control. Always present.
            var diaphragms = new List<int2>(stories);
            for (int l = 1; l < levels; l++)
                diaphragms.Add(new int2(l * 4 + 0, l * 4 + 2));
            Diaphragms = diaphragms.ToArray();
        }

        void Build()
        {
            if (built) arena.Dispose();
            arena = new Arena(Allocator.Persistent);

            BuildGeometry();

            int nb = Nodes.Length;
            int capHint = (Chords.Length + Rings.Length + Diagonals.Length + Diaphragms.Length + 4) * 27;
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
            foreach (var m in Diaphragms) AddBar(m.x, m.y);
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

            switch (preconditioner)
            {
                case Preconditioner.BlockJacobi: mJacobi = arena.floatBlockJacobi(in A); break;
                case Preconditioner.SSOR:        mSSOR = new floatSSOR(in A, ref arena); break;
                default:                         mIC0 = arena.floatIC0(in A); break;
            }
            cache = arena.floatLOBPCGCache(N, K);
            residX = arena.floatVec(N);
            residAx = arena.floatVec(N);

            built = true;
            justBuilt = true;
            builtEA = stiffnessEA;
            builtStories = stories;
            builtBraces = (bool[])braceOn.Clone();
            builtPrecond = preconditioner;
        }

        void Update()
        {
            if (stories != builtStories)
                braceOn = NewBraceArray(stories, braceOn);

            bool dirty = stories != builtStories || builtEA != stiffnessEA || preconditioner != builtPrecond;
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
            // iteration count collapses to a handful at steady state. The cold count -- the solve
            // on the frame right after a rebuild, when the cache is fresh -- is where the
            // preconditioners visibly differ, so latch it for display.
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
            // concentrates stress -- where the structure yields/buckles first. Eigenvectors are
            // unit-normalized, so only the RELATIVE distribution matters: normalize to the per-mode peak.
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
                foreach (var m in Chords) maxF = math.max(maxF, math.abs(Force(m.x, m.y)));
                foreach (var m in Rings) maxF = math.max(maxF, math.abs(Force(m.x, m.y)));
                foreach (var m in Diaphragms) maxF = math.max(maxF, math.abs(Force(m.x, m.y)));
                for (int s = 0; s < stories; s++)
                    if (braceOn[s])
                        for (int f = 0; f < 4; f++)
                        { var m = Diagonals[s * 4 + f]; maxF = math.max(maxF, math.abs(Force(m.x, m.y))); }
            }

            // blue (low) -> cyan -> green -> yellow -> red (peak) via hue 0.66..0.
            void Draw(int a, int b, Color baseColor)
            {
                Gizmos.color = stress
                    ? Color.HSVToRGB((1f - math.saturate(math.abs(Force(a, b)) / maxF)) * 0.66f, 0.9f, 1f)
                    : baseColor;
                Gizmos.DrawLine(P(a), P(b));
            }

            foreach (var m in Chords) Draw(m.x, m.y, Color.white);
            foreach (var m in Rings) Draw(m.x, m.y, Color.white);
            foreach (var m in Diaphragms) Draw(m.x, m.y, new Color(0.4f, 0.6f, 1f));
            for (int s = 0; s < stories; s++)
                if (braceOn[s])
                    for (int f = 0; f < 4; f++)
                    { var m = Diagonals[s * 4 + f]; Draw(m.x, m.y, Color.yellow); }

            Gizmos.color = Color.red;
            for (int c = 0; c < 4; c++) Gizmos.DrawSphere(P(c), 0.06f);
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 460, 380), GUI.skin.box);
            GUILayout.Label($"3D truss tower — LOBPCG k={K} over {N}-dof BSR (3x3 blocks), {frameMs:F2} ms/frame");
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
                // A stable slender tower has a genuinely SMALL softest eigenvalue (its lateral sway
                // mode), so an absolute lambda bar can't separate "soft but stable" from "mechanism".
                // The robust test is the softest mode's own residual ||A x0 - lambda0 x0|| / ||A x0||:
                // near zero for a genuine eigenpair, near one for a mechanism the solver cannot pin --
                // independent of tower height and of HOW MANY mechanism modes exist.
                bool unstable = !(softestResidual <= 0.25f);   // NaN-safe
                if (stabilityLabelStyle == null) stabilityLabelStyle = new GUIStyle(GUI.skin.label);
                stabilityLabelStyle.normal.textColor = unstable ? Color.red : Color.green;
                GUILayout.Label(unstable
                    ? $"softest mode is a mechanism (residual {softestResidual:F2}) — UNSTABLE"
                    : $"stable — softest sway lambda0={lambda[0]:E2}, residual {softestResidual:F3}", stabilityLabelStyle);
            }

            colorByStress = GUILayout.Toggle(colorByStress, "colour members by modal stress (red = breaks first)");

            GUILayout.Label("Diagonal bracing (per story):");
            for (int row = 0; row < braceOn.Length; row += 8)
            {
                GUILayout.BeginHorizontal();
                for (int i = row; i < math.min(row + 8, braceOn.Length); i++)
                    braceOn[i] = GUILayout.Toggle(braceOn[i], $"s{i}");
                GUILayout.EndHorizontal();
            }

            string modeTag = (built && lambda.IsCreated)
                ? $"lambda={Lam(lambda[shownMode])}{(shownModeResidual > 0.25f ? ", mechanism — shape undefined" : "")}"
                : "lambda=0";
            shownMode = (int)LabeledSlider($"mode {shownMode} ({modeTag})", shownMode, 0, 3.49f);
            stiffnessEA = LabeledSlider($"EA {stiffnessEA:F1}", stiffnessEA, 0.5f, 20f);
            modeAmplitude = LabeledSlider($"amplitude {modeAmplitude:F2}", modeAmplitude, 0f, 0.5f);
            stories = (int)LabeledSlider($"stories {stories}", stories, 1, 24.49f);
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

    /// <summary>Warm LOBPCG smallest-k eigenpairs of the tower stiffness matrix with an SSOR
    /// preconditioner (omega=1, symmetric Gauss-Seidel). Like IC(0) it carries inter-story
    /// coupling through its forward/backward sweeps, but as a stationary iteration rather than a
    /// factorization it needs roughly twice IC(0)'s iterations on this pencil.</summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct TrussEigenJobSSOR : IJob
    {
        [ReadOnly] public floatBSROperator Op;
        public floatSSOR Precond;   // not [ReadOnly]: SSOR.Apply writes its internal sweep scratch
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
