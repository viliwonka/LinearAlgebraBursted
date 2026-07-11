using System.Diagnostics;
using LinearAlgebra;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;

namespace LinearAlgebraDemos
{
    /// <summary>
    /// L1 (LAD) vs L2 robustness, re-solved every frame in one Burst job. A noisy
    /// animated plane gets a slider-controlled fraction of upward-biased outliers;
    /// the L2 plane (red, QR) is dragged up by them while the L1 plane (green,
    /// LP.lad) stays on the true surface. A tau slider switches the L1 fit to
    /// quantile regression (LP.ladBR/ladFN).
    /// </summary>
    public class LadFitDemo : MonoBehaviour
    {
        [Range(16, 2048)] public int pointCount = 384;
        [Range(0f, 0.3f)] public float noiseSigma = 0.05f;
        [Range(0f, 0.6f)] public float outlierFraction = 0.25f;
        [Range(0f, 10f)] public float outlierScale = 3f;
        [Range(0.05f, 0.95f)] public float tau = 0.5f;
        public bool animate = true;

        NativeArray<float3> points;
        NativeArray<float> l2Coeffs;   // 3: a·x + b·z + c
        NativeArray<float> l1Coeffs;   // 3
        NativeArray<float> stats;      // [0] l2 rms, [1] l1 objective, [2] l1 iters, [3] l1 ok, [4] l2 ok
        float solveMs;
        uint frame;

        void OnEnable()
        {
            points = new NativeArray<float3>(pointCount, Allocator.Persistent);
            l2Coeffs = new NativeArray<float>(3, Allocator.Persistent);
            l1Coeffs = new NativeArray<float>(3, Allocator.Persistent);
            stats = new NativeArray<float>(5, Allocator.Persistent);
        }

        void OnDisable()
        {
            if (points.IsCreated) points.Dispose();
            if (l2Coeffs.IsCreated) l2Coeffs.Dispose();
            if (l1Coeffs.IsCreated) l1Coeffs.Dispose();
            if (stats.IsCreated) stats.Dispose();
        }

        void Update()
        {
            if (points.Length != pointCount)
            {
                points.Dispose();
                points = new NativeArray<float3>(pointCount, Allocator.Persistent);
            }

            var job = new LadFitJob
            {
                Points = points,
                L2Coeffs = l2Coeffs,
                L1Coeffs = l1Coeffs,
                Stats = stats,
                NoiseSigma = noiseSigma,
                OutlierFraction = outlierFraction,
                OutlierScale = outlierScale,
                Tau = tau,
                Time = animate ? UnityEngine.Time.time : 0f,
                Seed = 1u + frame++,
            };

            var sw = Stopwatch.StartNew();
            job.Run();
            sw.Stop();
            solveMs = (float)sw.Elapsed.TotalMilliseconds;
        }

        void OnDrawGizmos()
        {
            if (!Application.isPlaying || !points.IsCreated) return;

            Gizmos.color = new Color(1f, 0.6f, 0.1f, 0.9f);
            for (int i = 0; i < points.Length; i++)
                Gizmos.DrawCube((Vector3)points[i], Vector3.one * 0.02f);

            DrawPlane(l2Coeffs, new Color(1f, 0.25f, 0.25f));   // L2: dragged by outliers
            DrawPlane(l1Coeffs, new Color(0.25f, 1f, 0.25f));   // L1: robust
        }

        void DrawPlane(NativeArray<float> c, Color color)
        {
            Gizmos.color = color;
            const int G = 8;
            for (int gx = 0; gx <= G; gx++)
            {
                for (int gz = 0; gz < G; gz++)
                {
                    float x0 = -1f + 2f * gx / G, z0 = -1f + 2f * gz / G;
                    float z1 = -1f + 2f * (gz + 1) / G;
                    float Y(float x, float z) => c[0] * x + c[1] * z + c[2];
                    Gizmos.DrawLine(new Vector3(x0, Y(x0, z0), z0), new Vector3(x0, Y(x0, z1), z1));
                    Gizmos.DrawLine(new Vector3(z0, Y(z0, x0), x0), new Vector3(z1, Y(z1, x0), x0));
                }
            }
        }

        void OnGUI()
        {
            GUILayout.BeginArea(new Rect(10, 10, 360, 210), GUI.skin.box);
            GUILayout.Label($"LAD (L1, green) vs LS (L2, red) — m={pointCount}");
            GUILayout.Label($"both solves: {solveMs:F3} ms   L1 iters: {stats[2]:F0}   ok: L1={stats[3] == 1f} L2={stats[4] == 1f}");
            GUILayout.Label($"L2: y = {l2Coeffs[0]:F3}x + {l2Coeffs[1]:F3}z + {l2Coeffs[2]:F3}   rms={stats[0]:F4}");
            GUILayout.Label($"L1: y = {l1Coeffs[0]:F3}x + {l1Coeffs[1]:F3}z + {l1Coeffs[2]:F3}   sum|r|={stats[1]:F3}");
            GUILayout.Label($"outliers {outlierFraction:P0} (upward-biased ×{outlierScale:F1})   tau={tau:F2}");
            outlierFraction = GUILayout.HorizontalSlider(outlierFraction, 0f, 0.6f);
            tau = GUILayout.HorizontalSlider(tau, 0.05f, 0.95f);
            GUILayout.EndArea();
        }
    }

    /// <summary>Generates biased-outlier points, fits L2 (QR) and L1/quantile (LP), one Burst job.</summary>
    [BurstCompile(CompileSynchronously = true)]
    public struct LadFitJob : IJob
    {
        public NativeArray<float3> Points;
        public NativeArray<float> L2Coeffs;
        public NativeArray<float> L1Coeffs;
        public NativeArray<float> Stats;
        public float NoiseSigma, OutlierFraction, OutlierScale, Tau, Time;
        public uint Seed;

        public void Execute()
        {
            int m = Points.Length;
            const int n = 3;

            var rng = new Random(Seed * 2654435761u + 1u);
            var gauss = new floatGaussian(0f, NoiseSigma);

            float ca = 0.5f * math.sin(0.31f * Time);
            float cb = 0.5f * math.cos(0.23f * Time);
            float cc = 0.2f * math.sin(0.13f * Time);

            for (int i = 0; i < m; i++)
            {
                float px = rng.NextFloat(-1f, 1f);
                float pz = rng.NextFloat(-1f, 1f);
                float py = ca * px + cb * pz + cc + gauss.Next(ref rng);
                if (rng.NextFloat() < OutlierFraction)
                    py += math.abs(rng.NextFloat()) * OutlierScale;   // upward-biased
                Points[i] = new float3(px, py, pz);
            }

            var A = new floatMxN(m, n, Allocator.Temp);
            var b = new floatN(m, Allocator.Temp);
            for (int i = 0; i < m; i++)
            {
                A[i, 0] = Points[i].x; A[i, 1] = Points[i].z; A[i, 2] = 1f;
                b[i] = Points[i].y;
            }

            // L1 first — LP.lad/ladBR/ladFN preserve A and b (in parameters)...
            var x1 = new floatN(n, Allocator.Temp);
            LPInfo l1Info;
            double l1Obj;
            if (Tau > 0.499f && Tau < 0.501f)
                l1Info = LP.lad(in A, in b, ref x1, out l1Obj);
            else if (m <= 512)
                l1Info = LP.ladBR(in A, in b, Tau, ref x1, out l1Obj);
            else
                l1Info = LP.ladFN(in A, in b, Tau, ref x1, out l1Obj);

            // ...then L2, whose solveInPlace destroys them.
            var x2 = new floatN(n, Allocator.Temp);
            DirectSolveInfo l2Info = QR.solveInPlace(ref A, ref b, ref x2);

            for (int j = 0; j < n; j++) { L1Coeffs[j] = x1[j]; L2Coeffs[j] = x2[j]; }

            float ss = 0f;
            for (int i = 0; i < m; i++)
            {
                float e = x2[0] * Points[i].x + x2[1] * Points[i].z + x2[2] - Points[i].y;
                ss += e * e;
            }
            Stats[0] = math.sqrt(ss / m);
            Stats[1] = (float)l1Obj;
            Stats[2] = l1Info.iterations;
            Stats[3] = l1Info ? 1f : 0f;
            Stats[4] = l2Info ? 1f : 0f;
        }
    }
}
