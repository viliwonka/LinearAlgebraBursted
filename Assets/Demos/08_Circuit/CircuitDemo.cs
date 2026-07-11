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
    /// Transient simulation of an RC resistor grid via modified nodal analysis:
    /// backward-Euler each frame solves the symmetric INDEFINITE system
    /// [[G + C/h, E], [ET, 0]] (node voltages + source/ground currents) with
    /// ILU(0)-preconditioned BiCGStab over scalar-block BSR inside a Burst job.
    /// A sinusoidal (or DC) source drives one corner, ground pins the other;
    /// node voltages render as a live heatmap. Drag resistance/capacitance and
    /// watch the wavefront diffuse.
    /// </summary>
    public class CircuitDemo : MonoBehaviour
    {
        const int W = 12, Hn = 8;
        [Range(0.05f, 10f)] public float resistance = 1f;      // per grid edge (ohm)
        [Range(0.001f, 0.5f)] public float capacitance = 0.05f; // per node to ground (F)
        [Range(0f, 5f)] public float sourceAmplitude = 3f;
        [Range(0f, 3f)] public float sourceFrequency = 0.5f;   // Hz; 0 = DC
        const float H = 1f / 60f;

        int NodeCount => W * Hn;
        int Unknowns => NodeCount + 2;   // + source current, + ground current
        int SrcNode => 0;
        int GndNode => NodeCount - 1;

        Arena arena;
        floatBSR A;
        floatILU0 precond;
        bool built;
        float builtR, builtC;

        NativeArray<float> voltages;   // previous node voltages (+2 aux)
        NativeArray<float> outStats;   // [0] iters, [1] converged, [2] rnorm, [3] source current
        float frameMs;

        void OnEnable()
        {
            voltages = new NativeArray<float>(Unknowns, Allocator.Persistent);
            outStats = new NativeArray<float>(4, Allocator.Persistent);
            Build();
        }

        void OnDisable()
        {
            if (built) { arena.Dispose(); built = false; }
            if (voltages.IsCreated) voltages.Dispose();
            if (outStats.IsCreated) outStats.Dispose();
        }

        void Build()
        {
            if (built) arena.Dispose();
            arena = new Arena(Allocator.Persistent);

            int n = NodeCount, nu = Unknowns;
            float g = 1f / resistance;
            float ch = capacitance / H;

            var builder = new floatBSRBuilder(nu, nu, 1, 1, Allocator.Temp, n * 6);

            // every diagonal entry stored (ILU0 requires it; aux rows start at 0)
            for (int i = 0; i < nu; i++) builder.AddValue(i, i, 0f);

            for (int j = 0; j < Hn; j++)
                for (int i = 0; i < W; i++)
                {
                    int id = j * W + i;
                    builder.AddValue(id, id, ch);   // capacitor to ground
                    if (i + 1 < W) AddResistor(ref builder, id, id + 1, g);
                    if (j + 1 < Hn) AddResistor(ref builder, id, id + W, g);
                }

            // MNA rows: v_src = V(t), v_gnd = 0 (E columns/rows, zero diagonal)
            int rSrc = n, rGnd = n + 1;
            builder.AddValue(SrcNode, rSrc, 1f); builder.AddValue(rSrc, SrcNode, 1f);
            builder.AddValue(GndNode, rGnd, 1f); builder.AddValue(rGnd, GndNode, 1f);

            A = builder.ToBSR(ref arena);   // full storage: BiCGStab path, no symmetry claim
            builder.Dispose();
            precond = arena.floatILU0(in A);

            built = true; builtR = resistance; builtC = capacitance;
        }

        static void AddResistor(ref floatBSRBuilder b, int p, int q, float g)
        {
            b.AddValue(p, p, g); b.AddValue(q, q, g);
            b.AddValue(p, q, -g); b.AddValue(q, p, -g);
        }

        void Update()
        {
            if (builtR != resistance || builtC != capacitance) Build();

            float t = Time.time;
            float vSrc = sourceFrequency < 0.01f
                ? sourceAmplitude
                : sourceAmplitude * math.sin(2f * math.PI * sourceFrequency * t);

            var job = new CircuitStepJob
            {
                A = A, Precond = precond,
                Voltages = voltages, Out = outStats,
                NodeCount = NodeCount, CapOverH = capacitance / H,
                VSource = vSrc,
            };

            var sw = Stopwatch.StartNew();
            job.Run();
            sw.Stop();
            frameMs = (float)sw.Elapsed.TotalMilliseconds;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !voltages.IsCreated) return;
            for (int j = 0; j < Hn; j++)
                for (int i = 0; i < W; i++)
                {
                    float v = voltages[j * W + i];
                    float s = math.saturate(math.abs(v) / math.max(0.5f, sourceAmplitude));
                    Gizmos.color = v >= 0f
                        ? Color.Lerp(new Color(0.1f, 0.1f, 0.15f), Color.red, s)
                        : Color.Lerp(new Color(0.1f, 0.1f, 0.15f), Color.blue, s);
                    Gizmos.DrawCube(new Vector3(i * 0.4f, j * 0.4f, 0f),
                                    new Vector3(0.34f, 0.34f, 0.05f + 0.25f * s));
                }
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 420, 170), GUI.skin.box);
            GUILayout.Label($"RC grid MNA — {Unknowns} unknowns, ILU(0)-BiCGStab, {frameMs:F2} ms/frame");
            GUILayout.Label($"iters: {outStats[0]:F0}   converged: {outStats[1] == 1f}   rnorm: {outStats[2]:E1}");
            GUILayout.Label($"source current: {outStats[3]:F2} A   V(src)={voltages[SrcNode]:F2}");
            resistance = LabeledSlider($"R {resistance:F2} ohm", resistance, 0.05f, 10f);
            capacitance = LabeledSlider($"C {capacitance:F3} F", capacitance, 0.001f, 0.5f);
            sourceFrequency = LabeledSlider($"f {sourceFrequency:F2} Hz", sourceFrequency, 0f, 3f);
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

    /// <summary>One backward-Euler MNA step: rhs from previous voltages, BiCGStab solve, write back.</summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct CircuitStepJob : IJob
    {
        [ReadOnly] public floatBSR A;
        [ReadOnly] public floatILU0 Precond;
        public NativeArray<float> Voltages;
        public NativeArray<float> Out;
        public int NodeCount;
        public float CapOverH, VSource;

        public void Execute()
        {
            int nu = Voltages.Length;

            var b = new floatN(nu, Allocator.Temp);
            for (int i = 0; i < NodeCount; i++)
                b[i] = CapOverH * Voltages[i];
            b[NodeCount] = VSource;   // v_src constraint rhs
            b[NodeCount + 1] = 0f;    // v_gnd constraint rhs

            // x is a zero-copy VIEW over Voltages: the previous frame's solution is the
            // warm start as-is, and BiCGStab writes the new voltages straight back --
            // no boundary copies in either direction.
            var x = new floatN(Voltages);

            var op = new floatBSROperator(in A);
            var r = new floatN(nu, Allocator.Temp);
            var rHat0 = new floatN(nu, Allocator.Temp);
            var p = new floatN(nu, Allocator.Temp);
            var v = new floatN(nu, Allocator.Temp);
            var t = new floatN(nu, Allocator.Temp);
            var pHat = new floatN(nu, Allocator.Temp);
            var sHat = new floatN(nu, Allocator.Temp);

            SolveInfo info = Krylov.pbiCGStab(in op, in Precond, in b, ref x,
                                              ref r, ref rHat0, ref p, ref v, ref t,
                                              ref pHat, ref sHat,
                                              400, 1e-6f);

            Out[0] = info.iterations;
            Out[1] = info ? 1f : 0f;
            Out[2] = (float)info.rnorm;
            Out[3] = Voltages[NodeCount];   // lambda_src = current through the source
        }
    }
}
