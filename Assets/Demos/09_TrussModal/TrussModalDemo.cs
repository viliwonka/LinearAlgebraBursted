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
    /// Vibration mode shapes of a small house-frame truss. Global stiffness A and a lumped
    /// diagonal mass matrix B are assembled once (2x2-block symmetric BSR, same node/member
    /// layout as <see cref="TrussStabilityDemo"/>); a single Burst job solves the generalized
    /// eigenproblem A*phi = lambda*B*phi via block-Jacobi-preconditioned LOBPCG for the
    /// <see cref="ModeCount"/> smallest pairs (natural frequencies omega = sqrt(lambda)). The
    /// solve runs once at Start (and again only if EA/mass sliders change) — every frame just
    /// displaces node positions sinusoidally along the selected mode shape.
    /// </summary>
    public class TrussModalDemo : MonoBehaviour
    {
        [Range(0.5f, 20f)] public float stiffnessEA = 8f;
        [Range(0.05f, 5f)] public float nodeMass = 1f;
        [Range(0f, 0.5f)] public float modeAmplitude = 0.15f;
        [Range(0.1f, 5f)] public float animationSpeed = 1f;
        [Range(0, ModeCount - 1)] public int shownMode;

        const int ModeCount = 5;

        // house frame: 4 bottom, 4 mid, 1 ridge -- same layout as TrussStabilityDemo
        static readonly float2[] Nodes =
        {
            new float2(0, 0), new float2(1, 0), new float2(2, 0), new float2(3, 0),
            new float2(0, 1), new float2(1, 1), new float2(2, 1), new float2(3, 1),
            new float2(1.5f, 1.9f),
        };
        static readonly int2[] Fixed =
        {
            new int2(0, 1), new int2(1, 2), new int2(2, 3),      // bottom chord
            new int2(4, 5), new int2(5, 6), new int2(6, 7),      // mid chord
            new int2(0, 4), new int2(1, 5), new int2(2, 6), new int2(3, 7),   // columns
            new int2(4, 8), new int2(5, 8), new int2(6, 8), new int2(7, 8),   // roof
        };
        static readonly int2[] Braces =
        {
            new int2(0, 5), new int2(1, 6), new int2(2, 7),
            new int2(1, 4), new int2(2, 5), new int2(3, 6),
        };

        int N => Nodes.Length * 2;

        Arena arena;
        floatBSR A;      // stiffness
        floatBSR B;      // lumped mass
        floatBlockJacobi precond;
        floatLOBPCGCache cache;
        floatN lambda;      // arena-owned view of cache.lambda after solve
        floatMxN modes;     // arena-owned view of cache.X (ModeCount x N)
        bool built;
        bool solved;
        float builtEA;
        float builtMass;
        NativeArray<float> outStats;   // [0] iterations, [1] converged, [2] maxResidual

        void Start()
        {
            outStats = new NativeArray<float>(3, Allocator.Persistent);
            Build();
            Solve();
        }

        void OnDestroy()
        {
            if (built) { arena.Dispose(); built = false; }
            if (outStats.IsCreated) outStats.Dispose();
        }

        void Build()
        {
            if (built) arena.Dispose();
            arena = new Arena(Allocator.Persistent);

            int nb = Nodes.Length;
            var kBuilder = new floatBSRBuilder(nb, nb, 2, 2, Allocator.Temp, 64);
            var mBuilder = new floatBSRBuilder(nb, nb, 2, 2, Allocator.Temp, nb);

            void AddBar(int a, int b)
            {
                float2 d = Nodes[b] - Nodes[a];
                float L = math.length(d);
                float2 u = d / L;
                float k = stiffnessEA / L;
                int lo = math.min(a, b), hi = math.max(a, b);
                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 2; c++)
                    {
                        float v = k * u[r] * u[c];
                        kBuilder.AddValue(2 * a + r, 2 * a + c, v);
                        kBuilder.AddValue(2 * b + r, 2 * b + c, v);
                        kBuilder.AddValue(2 * hi + r, 2 * lo + c, -v);
                    }
            }

            foreach (var m in Fixed) AddBar(m.x, m.y);
            foreach (var m in Braces) AddBar(m.x, m.y);

            // pinned supports at nodes 0 and 3: penalty on their diagonal blocks, within ~3
            // decades of the bar stiffness (see TrussStabilityDemo.Build) -- the resulting
            // frequency at these dof is pushed far above the physical spectrum rather than
            // exactly removed, so it does not compete for the ModeCount smallest pairs.
            for (int d = 0; d < 2; d++)
            {
                kBuilder.AddValue(0 + d, 0 + d, 1e3f);
                kBuilder.AddValue(6 + d, 6 + d, 1e3f);   // node 3 -> dof 6,7
            }

            // lumped diagonal mass at every node (translational dof only -- pin-jointed truss
            // members carry no rotational dof).
            for (int i = 0; i < nb; i++)
                for (int d = 0; d < 2; d++)
                    mBuilder.AddValue(2 * i + d, 2 * i + d, nodeMass);

            A = kBuilder.ToBSRSymmetric(ref arena);
            kBuilder.Dispose();
            B = mBuilder.ToBSRSymmetric(ref arena);
            mBuilder.Dispose();

            precond = arena.floatBlockJacobi(in A);
            cache = arena.floatLOBPCGCache(N, ModeCount);

            built = true;
            builtEA = stiffnessEA;
            builtMass = nodeMass;
        }

        void Solve()
        {
            var job = new TrussModalJob
            {
                A = new floatBSROperator(in A),
                B = new floatBSROperator(in B),
                Precond = precond,
                Cache = cache,
                Out = outStats,
                K = ModeCount,
            };
            IJobExtensions.RunByRef(ref job);
            cache = job.Cache;

            solved = outStats[1] == 1f;
            if (!solved)
                Debug.LogWarning($"TrussModalDemo: LOBPCG did not converge (iterations={outStats[0]}, maxResidual={outStats[2]:E2}) -- mode shapes not animated.");

            lambda = cache.lambda;
            modes = cache.X;
        }

        void Update()
        {
            // Solve once at Start and again only when a parameter that changes A/B is edited --
            // no per-frame solve. The sinusoidal displacement in OnDrawGizmos is the only
            // per-frame work and reads Time.time directly, so it needs no state here.
            if (builtEA != stiffnessEA || builtMass != nodeMass)
            {
                Build();
                Solve();
            }
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !built || !solved || !lambda.IsCreated) return;

            float omega = math.sqrt(math.max(0f, lambda[shownMode]));
            float wob = modeAmplitude * math.sin(omega * animationSpeed * Time.time);
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
            foreach (var m in Braces) Gizmos.DrawLine(P(m.x), P(m.y));
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(P(0), 0.06f); Gizmos.DrawSphere(P(3), 0.06f);
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 440, 260), GUI.skin.box);
            GUILayout.Label($"Truss modal analysis — LOBPCG k={ModeCount} generalized eigenproblem over {N}-dof BSR");
            GUILayout.Label($"converged: {solved}   iterations: {outStats[0]:F0}   maxResidual: {outStats[2]:E2}");
            if (built && solved && lambda.IsCreated)
            {
                float omega = math.sqrt(math.max(0f, lambda[shownMode]));
                float freqHz = omega / (2f * math.PI);
                GUILayout.Label($"mode {shownMode}:  omega = {omega:F3} rad/s   f = {freqHz:F3} Hz");
            }
            else
            {
                GUILayout.Label("solver has not produced a usable mode shape yet");
            }

            shownMode = (int)LabeledSlider($"mode {shownMode}", shownMode, 0, ModeCount - 0.51f);
            stiffnessEA = LabeledSlider($"EA {stiffnessEA:F1}", stiffnessEA, 0.5f, 20f);
            nodeMass = LabeledSlider($"node mass {nodeMass:F2}", nodeMass, 0.05f, 5f);
            modeAmplitude = LabeledSlider($"amplitude {modeAmplitude:F2}", modeAmplitude, 0f, 0.5f);
            animationSpeed = LabeledSlider($"anim speed {animationSpeed:F2}", animationSpeed, 0.1f, 5f);
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

    /// <summary>
    /// Generalized (mass-matrix) LOBPCG solve of the ModeCount smallest pairs of A*phi =
    /// lambda*B*phi, block-Jacobi-preconditioned from the stiffness matrix A.
    /// </summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct TrussModalJob : IJob
    {
        [ReadOnly] public floatBSROperator A;
        [ReadOnly] public floatBSROperator B;
        [ReadOnly] public floatBlockJacobi Precond;
        public floatLOBPCGCache Cache;
        public NativeArray<float> Out;
        public int K;

        public void Execute()
        {
            // Default tolerance on purpose — same spurious-Ritz-collapse rationale as
            // TrussStabilityDemo.TrussEigenJob (float LOBPCG on a penalty-conditioned pencil
            // degrades past the default tolerance).
            LOBPCGInfo info = Eigen.lobpcg(in A, in B, in Precond, ref Cache, K);
            Out[0] = info.iterations;
            Out[1] = info ? 1f : 0f;
            Out[2] = (float)info.maxResidual;
        }
    }
}
